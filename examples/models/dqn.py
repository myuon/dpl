"""DQN Agent implementation."""

import numpy as np

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.core import as_jax
from dpl.optimizers import Adam
from dpl.agent import ReplayBuffer, BaseAgent


class DQNAgent(BaseAgent):
    """DQN Agent

    - Experience Replay
    - epsilon-greedy探索
    - target_qnet指定時: Double DQN (action選択はqnet、Q値評価はtarget_qnet)
    - target_qnet省略時: 通常DQN (qnetでmax Q値を直接使用)
    """

    def __init__(
        self,
        qnet: L.Layer,
        action_size: int,
        target_qnet: L.Layer | None = None,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 10000,
        tau: float = 0.005,
        warmup_steps: int = 500,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.warmup_steps = warmup_steps

        # Q-Network (外部から注入)
        self.qnet = qnet
        self.target_qnet = target_qnet  # Noneなら通常DQN

        # Target networkを初期化 (Double DQNの場合のみ)
        if self.target_qnet is not None:
            self._hard_update_target()

        # Optimizer
        self.optimizer = Adam(lr=lr).setup(self.qnet)

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size)

        # 学習ステップカウンタ
        self.learn_step = 0

        # モニタリング用
        self.last_loss: float | None = None

    def _soft_update_target(self):
        """Target networkをsoft update: θ' ← τθ + (1-τ)θ'"""
        if self.target_qnet is None:
            return
        for main_param, target_param in zip(
            self.qnet.params(), self.target_qnet.params()
        ):
            if main_param.data is not None and target_param.data is not None:
                target_param.data = (
                    self.tau * main_param.data + (1 - self.tau) * target_param.data
                )

    def _hard_update_target(self):
        """Target networkをメインnetworkで完全に同期"""
        if self.target_qnet is None:
            return
        for main_param, target_param in zip(
            self.qnet.params(), self.target_qnet.params()
        ):
            if main_param.data is not None:
                target_param.data = main_param.data.copy()

    def _greedy_action(self, state: np.ndarray) -> int:
        """Q値が最大のアクションを選択"""
        # バッチ次元を追加 (MLP: 1D→2D, CNN: 3D→4D)
        state_v = Variable(state[np.newaxis, ...].astype(np.float32))
        q_values = self.qnet(state_v)
        return int(np.argmax(q_values.data_required))

    def get_action(self, state: np.ndarray) -> int:
        """epsilon-greedy戦略でアクションを選択"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return self._greedy_action(state)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """アクションを選択"""
        if explore:
            return self.get_action(state)
        return self._greedy_action(state)

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
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict | None:
        """Replay Bufferからサンプリングして更新"""
        # Warmup期間中は更新しない
        if len(self.buffer) < self.warmup_steps:
            return None

        # バッファサイズが足りない場合は更新しない
        if len(self.buffer) < self.batch_size:
            return None

        # ミニバッチサンプリング
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # GPU計算のためJAX配列に変換
        states_v = Variable(as_jax(states))
        next_states_v = Variable(as_jax(next_states))

        # 次状態のQ値を計算 (勾配不要)
        next_q_values = self.qnet(next_states_v).data_required

        if self.target_qnet is not None:
            # Double DQN: action選択はqnet、Q値評価はtarget_qnet
            next_actions = np.argmax(next_q_values, axis=1)
            next_q_target = self.target_qnet(next_states_v).data_required
            max_next_q = next_q_target[np.arange(self.batch_size), next_actions]
        else:
            # 通常DQN: qnetでmax Q値を直接使用
            max_next_q = np.max(next_q_values, axis=1)

        # TD target: r + γ * max_a Q(s', a)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # 現在のQ値を計算（勾配あり）
        q_values = self.qnet(states_v)  # (batch, action_size)

        # one-hotマスクでアクションのQ値を取り出す（勾配を維持）
        action_masks = np.eye(self.action_size)[actions.astype(np.int64)]
        current_q = F.sum(
            q_values * Variable(as_jax(action_masks.astype(np.float32))), axis=1
        )  # (batch,)

        # MSE Loss
        targets_v = Variable(as_jax(targets.astype(np.float32)))
        loss = F.mean_squared_error(current_q, targets_v)

        # 勾配計算と更新
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        # Target networkのsoft update
        self.learn_step += 1
        self._soft_update_target()

        self.last_loss = float(loss.data_required)
        return {"loss": self.last_loss}

    def decay_epsilon(self):
        """エピソード終了時にepsilonを減衰"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def dqn_stats_extractor(agent) -> str | None:
    """DQN Agent用の統計抽出関数"""
    epsilon = getattr(agent, "epsilon", None)
    loss = getattr(agent, "last_loss", None)

    parts = []
    if epsilon is not None:
        parts.append(f"ε={epsilon:.3f}")
    if loss is not None:
        parts.append(f"L={loss:.4f}")

    if not parts:
        return None
    return ", " + ", ".join(parts)
