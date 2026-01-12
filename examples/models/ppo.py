"""PPO (Proximal Policy Optimization) Agent implementation.

On-policy algorithm with:
- Actor-Critic architecture
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
"""

import numpy as np

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent import BaseAgent


def log_softmax(logits: Variable, axis: int = 1) -> Variable:
    """log(softmax(x)) を数値的に安定に計算"""
    x_max = F.max(logits, axis=axis, keepdims=True)
    x_shifted = logits - x_max
    log_sum_exp = F.log(F.sum(F.exp(x_shifted), axis=axis, keepdims=True))
    return x_shifted - log_sum_exp


class RolloutBuffer:
    """Trajectory収集用バッファ（On-policy: 使い捨て）

    PPO用: 一定ステップ収集 → GAE計算 → 更新 → クリア
    """

    def __init__(self, buffer_size: int, obs_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim

        self.obs = np.zeros((buffer_size, obs_dim), np.float32)
        self.actions = np.zeros(buffer_size, np.int64)
        self.rewards = np.zeros(buffer_size, np.float32)
        self.dones = np.zeros(buffer_size, np.float32)
        self.log_probs = np.zeros(buffer_size, np.float32)
        self.values = np.zeros(buffer_size, np.float32)
        self.advantages = np.zeros(buffer_size, np.float32)
        self.returns = np.zeros(buffer_size, np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            log_prob: float, value: float):
        """1ステップ分のデータを追加"""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """GAE (Generalized Advantage Estimation) を計算

        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        A_t = Σ (γλ)^k δ_{t+k}
        """
        size = self.ptr

        # next_values: 次状態の価値（最後はlast_value）
        next_values = np.zeros(size, np.float32)
        next_values[:-1] = self.values[1:size]
        next_values[-1] = last_value

        # GAE計算（逆順で累積）
        gae = 0.0
        for t in reversed(range(size)):
            # 終端状態では次状態の価値は0
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values[t] * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Returns = Advantages + Values
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_batches(self, batch_size: int):
        """ミニバッチでイテレート（シャッフル）"""
        size = self.ptr
        indices = np.random.permutation(size)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield (
                self.obs[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )

    def clear(self):
        """バッファをクリア（update後に呼ぶ）"""
        self.ptr = 0
        self.full = False


class ActorCritic(L.Layer):
    """Actor-Critic共有ネットワーク（離散行動）

    Actor: π(a|s) - 行動確率を出力
    Critic: V(s) - 状態価値を出力
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 共有層
        self.shared = L.Sequential(
            L.Linear(hidden_size, in_size=obs_dim),
            F.relu,
            L.Linear(hidden_size),
            F.relu,
        )
        # Actor head: action logits
        self.actor_head = L.Linear(action_dim, in_size=hidden_size)
        # Critic head: state value
        self.critic_head = L.Linear(1, in_size=hidden_size)

    def forward(self, *inputs: Variable) -> tuple[Variable, Variable]:
        """Returns: (action_logits, value)"""
        obs = inputs[0]
        h = self.shared(obs)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return logits, value

    def get_action_and_value(self, obs: np.ndarray) -> tuple[int, float, float]:
        """行動サンプリング + log_prob + value

        Args:
            obs: (obs_dim,) の観測

        Returns:
            action: 選択された行動
            log_prob: その行動のlog確率
            value: 状態価値
        """
        logits, value = self(Variable(obs[None].astype(np.float32)))
        probs = F.softmax(logits)
        probs_data = probs.data_required[0]

        # Categorical sampling
        action = np.random.choice(self.action_dim, p=probs_data)
        log_prob = float(np.log(probs_data[action] + 1e-8))

        return action, log_prob, float(value.data_required[0, 0])

    def get_value(self, obs: np.ndarray) -> float:
        """状態価値のみ取得"""
        _, value = self(Variable(obs[None].astype(np.float32)))
        return float(value.data_required[0, 0])

    def evaluate_actions(
        self, obs: Variable, actions: np.ndarray
    ) -> tuple[Variable, Variable, Variable]:
        """バッチ評価: log_prob, entropy, value

        Args:
            obs: (batch, obs_dim) の観測
            actions: (batch,) の行動

        Returns:
            action_log_probs: 選択された行動のlog確率
            entropy: エントロピー
            values: 状態価値
        """
        logits, values = self(obs)
        probs = F.softmax(logits)
        log_probs = log_softmax(logits)

        # 選択されたアクションのlog_prob
        action_masks = np.eye(self.action_dim)[actions.astype(int)]
        action_log_probs = F.sum(log_probs * Variable(action_masks.astype(np.float32)), axis=1)

        # Entropy: -Σ p log p
        entropy = -F.sum(probs * log_probs, axis=1)

        return action_log_probs, entropy, values


class PPOAgent(BaseAgent):
    """PPO Agent (Clipped版)

    On-policy algorithm:
    1. 現在のポリシーでrollout_stepsステップ収集
    2. GAEでadvantageを計算
    3. n_epochs回、ミニバッチで更新
    4. バッファをクリアして1に戻る
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        action_size: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3e-4,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        max_grad_norm: float = 0.5,
    ):
        self.actor_critic = actor_critic
        self.action_size = action_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm

        self.optimizer = Adam(lr=lr).setup(self.actor_critic)
        self.buffer = RolloutBuffer(rollout_steps, obs_dim)

        # 内部状態
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_obs: np.ndarray | None = None

        # モニタリング用
        self.last_policy_loss: float | None = None
        self.last_value_loss: float | None = None
        self.last_entropy: float | None = None

    def get_action(self, state: np.ndarray) -> int:
        """行動を選択（常に確率的）"""
        action, log_prob, value = self.actor_critic.get_action_and_value(state)
        self._last_log_prob = log_prob
        self._last_value = value
        self._last_obs = state
        return action

    def _greedy_action(self, state: np.ndarray) -> int:
        """Greedy行動（argmax）- 通常はPPOには使わない"""
        logits, _ = self.actor_critic(Variable(state[None].astype(np.float32)))
        return int(np.argmax(logits.data_required[0]))

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """アクションを選択

        PPOは確率的ポリシーなので、explore=Falseでも確率的サンプリングを使用。
        (DQNのようなargmax評価はPPOには不適切)

        Args:
            state: 観測
            explore: Trueなら内部状態も更新（学習用）、Falseなら行動のみ返す（評価用）
        """
        if explore:
            return self.get_action(state)
        else:
            # 評価時も確率的サンプリング（ただし内部状態は更新しない）
            logits, _ = self.actor_critic(Variable(state[None].astype(np.float32)))
            probs = F.softmax(logits).data_required[0]
            return int(np.random.choice(self.action_size, p=probs))

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        terminated=None,
    ):
        """経験をバッファに保存"""
        self.buffer.add(
            state, action, reward, done,
            self._last_log_prob, self._last_value
        )
        self._last_obs = next_state

    def update(self) -> dict | None:
        """rollout_steps分たまったらupdate"""
        if not self.buffer.full:
            return None

        # 最後の状態の価値を取得（ブートストラップ用）
        if self._last_obs is not None:
            last_value = self.actor_critic.get_value(self._last_obs)
        else:
            last_value = 0.0

        # GAE計算
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # n_epochs × ミニバッチ更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                loss_dict = self._update_batch(batch)
                total_policy_loss += loss_dict["policy_loss"]
                total_value_loss += loss_dict["value_loss"]
                total_entropy += loss_dict["entropy"]
                n_updates += 1

        # バッファクリア
        self.buffer.clear()

        # 平均を記録
        self.last_policy_loss = total_policy_loss / n_updates
        self.last_value_loss = total_value_loss / n_updates
        self.last_entropy = total_entropy / n_updates

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
        }

    def _update_batch(self, batch: tuple) -> dict:
        """1ミニバッチの更新"""
        obs, actions, old_log_probs, advantages, returns = batch

        # Advantage正規化（重要！）
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std

        # 現在のポリシーで評価
        new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
            Variable(obs.astype(np.float32)), actions
        )

        # Policy loss (PPO Clipped)
        ratio = F.exp(new_log_probs - Variable(old_log_probs.astype(np.float32)))
        adv_v = Variable(advantages.astype(np.float32))

        surr1 = ratio * adv_v

        # Clipped ratio（勾配を止める）
        ratio_data = ratio.data_required
        clipped_ratio = np.clip(ratio_data, 1 - self.clip_eps, 1 + self.clip_eps)
        surr2 = Variable(clipped_ratio.astype(np.float32)) * adv_v

        # min(surr1, surr2) を要素ごとに計算
        surr1_data = surr1.data_required
        surr2_data = surr2.data_required
        min_surr_data = np.minimum(surr1_data, surr2_data)

        # どちらがminかで勾配を選択
        use_surr1 = (surr1_data <= surr2_data).astype(np.float32)
        policy_loss_per_sample = surr1 * Variable(use_surr1) + surr2 * Variable(1 - use_surr1)
        policy_loss = -F.sum(policy_loss_per_sample) / len(actions)

        # Value loss
        values_flat = F.reshape(values, (-1,))
        value_loss = F.mean_squared_error(values_flat, Variable(returns.astype(np.float32)))

        # Entropy bonus（探索促進）
        entropy_mean = F.sum(entropy) / len(actions)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

        # 更新
        self.actor_critic.cleargrads()
        loss.backward()

        # Gradient clipping
        self._clip_grad_norm()

        self.optimizer.update()

        return {
            "policy_loss": float(policy_loss.data_required),
            "value_loss": float(value_loss.data_required),
            "entropy": float(entropy_mean.data_required),
        }

    def _clip_grad_norm(self):
        """Gradient clipping (global norm)"""
        grads = []
        for param in self.actor_critic.params():
            if param.grad is not None:
                grads.append(param.grad.data_required.flatten())

        if not grads:
            return

        all_grads = np.concatenate(grads)
        global_norm = np.sqrt(np.sum(all_grads**2))

        if global_norm > self.max_grad_norm:
            scale = self.max_grad_norm / global_norm
            for param in self.actor_critic.params():
                if param.grad is not None:
                    param.grad.data = param.grad.data_required * scale

    def decay_epsilon(self):
        """PPOはepsilon-greedyを使わないのでno-op"""
        pass


def ppo_stats_extractor(agent) -> str | None:
    """PPO Agent用の統計抽出関数"""
    policy_loss = getattr(agent, "last_policy_loss", None)
    value_loss = getattr(agent, "last_value_loss", None)
    entropy = getattr(agent, "last_entropy", None)

    parts = []
    if policy_loss is not None:
        parts.append(f"π={policy_loss:.4f}")
    if value_loss is not None:
        parts.append(f"V={value_loss:.4f}")
    if entropy is not None:
        parts.append(f"H={entropy:.3f}")

    if not parts:
        return None
    return ", " + ", ".join(parts)
