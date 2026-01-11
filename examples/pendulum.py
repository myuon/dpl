# %% [markdown]
# # Pendulum-v1 強化学習
#
# SAC（Soft Actor-Critic）による連続行動空間の学習
# dpl（自作NNライブラリ）を使用

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from dpl import Variable
import dpl.functions as F
import dpl.layers as L
from dpl.optimizers import Adam

from dpl.agent import BaseAgent, ReplayBuffer
from dpl.agent_trainer import AgentTrainer, GymEnvWrapper, EvalResult

# %% [markdown]
# ## 環境仕様（参照用）
#
# | 項目 | 内容 |
# |------|------|
# | 観測 | shape=(3,), [cos(θ), sin(θ), angular_velocity] |
# | 観測範囲 | [-1, -1, -8] ~ [1, 1, 8] |
# | 行動 | shape=(1,), 連続値トルク |
# | 行動範囲 | [-2, 2] |
# | 報酬 | 常に負（-16.27 ~ 0）、振り子が上向きで0に近づく |
# | terminated | 常にFalse |
# | truncated | 200ステップでTrue |

# %% [markdown]
# ## SAC（Soft Actor-Critic）
#
# 最大エントロピー強化学習。探索を"ノイズで足す"から"目的関数で制御する"へ。
#
# ## SACの特徴（TD3からの改良点）
# 1. **確率方策（Stochastic Policy）**: π(a|s) = tanh(μ(s) + σ(s) * ε)
#    → 探索がポリシー自体に組み込まれる
# 2. **エントロピー正則化**: 目的 = E[Q(s,a) - α * log π(a|s)]
#    → "高得点を取りつつ、不確実性も残す"
# 3. **Reparameterization Trick**: a = tanh(μ + σ * ε)で勾配を流す
# 4. **自動温度調整**: αを学習して目標エントロピーを維持
#
# ## 更新式
# - Target: y = r + γ * (min(Q1'(s',a'), Q2'(s',a')) - α * log π(a'|s'))
# - Critic: L = E[(Q(s,a) - y)²]
# - Actor: L = E[α * log π(a|s) - Q(s, a)]  （エントロピー付き方策勾配）
# - α: L = -α * E[log π(a|s) + target_entropy]


# %%
from dpl.layers import Layer, Sequential


class GaussianPolicy(Layer):
    """ガウス方策ネットワーク（SAC Actor）

    状態 s → (μ, log_σ) を出力し、reparameterization trickで行動をサンプル
    a = tanh(μ + σ * ε), ε ~ N(0, I)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 2.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共有層
        self.shared = Sequential(
            L.Linear(hidden_dim, in_size=state_dim),
            F.relu,
            L.Linear(hidden_dim),
            F.relu,
        )

        # μ と log_σ を別々に出力
        self.mean_head = L.Linear(action_dim)
        self.log_std_head = L.Linear(action_dim)

    def predict(self, state: Variable) -> tuple[Variable, Variable]:
        """状態から平均と標準偏差を出力"""
        h = self.shared.apply(state)

        mean = self.mean_head.apply(h)
        log_std = self.log_std_head.apply(h)

        # log_std をクランプ（数値安定性）
        log_std_data = np.clip(
            log_std.data_required, self.log_std_min, self.log_std_max
        )
        log_std = Variable(log_std_data)

        return mean, log_std

    def sample(
        self, state: Variable
    ) -> tuple[Variable, Variable, Variable, Variable]:
        """行動をサンプル（reparameterization trick）

        Returns:
            action: tanh-squashed行動（スケール済み）
            log_prob: log π(a|s)（tanh補正込み）
            mean: 方策の平均（評価時用）
            log_std: 方策のlog標準偏差
        """
        mean, log_std = self.predict(state)
        std = F.exp(log_std)

        # Reparameterization trick: z = μ + σ * ε
        eps = np.random.randn(*mean.shape).astype(np.float32)
        z = mean + std * Variable(eps)

        # tanh-squash
        action = F.tanh(z) * self.action_scale

        # log π(a|s) の計算（tanh補正込み）
        # log π = log N(z; μ, σ) - Σ log(1 - tanh²(z))
        log_prob = self._compute_log_prob(z, mean, log_std)

        return action, log_prob, mean, log_std

    def _compute_log_prob(
        self, z: Variable, mean: Variable, log_std: Variable
    ) -> Variable:
        """log π(a|s) を計算（tanh補正込み）"""
        std = F.exp(log_std)

        # ガウス分布のlog確率
        # log N(z; μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((z-μ)/σ)²
        diff = z - mean
        log_prob = (
            -0.5 * np.log(2 * np.pi)
            - log_std
            - 0.5 * (diff / std) ** 2
        )

        # tanh補正: -log(1 - tanh²(z)) = -log(1 - a²/scale²)
        # 数値安定性のため、直接tanh(z)から計算
        tanh_z = F.tanh(z)
        # log(1 - tanh²(z) + ε)
        log_det = F.log(
            Variable(np.ones_like(z.data_required, dtype=np.float32))
            - tanh_z * tanh_z
            + Variable(np.full_like(z.data_required, 1e-6, dtype=np.float32))
        )
        log_prob = log_prob - log_det

        # 行動次元で合計
        log_prob = log_prob.sum(axis=1, keepdims=True)
        return log_prob


# %%
class QNetwork(Layer):
    """Q関数ネットワーク（SAC Critic）

    (状態 s, 行動 a) → Q(s, a) を出力
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # 状態と行動を結合して入力
        self.fc1 = L.Linear(hidden_dim, in_size=state_dim + action_dim)
        self.fc2 = L.Linear(hidden_dim)
        self.fc3 = L.Linear(1)

    def predict(self, state: Variable, action: Variable) -> Variable:
        """状態と行動からQ値を出力"""
        # 状態と行動を結合（axis=1で特徴量方向に結合）
        x = F.concat([state, action], axis=1)
        x = F.relu(self.fc1.apply(x))
        x = F.relu(self.fc2.apply(x))
        return self.fc3.apply(x)


# %%
def soft_update(target: Layer, source: Layer, tau: float):
    """ターゲットネットワークのソフト更新: θ' ← τθ + (1-τ)θ'"""
    for target_param, source_param in zip(target.params(), source.params()):
        target_param.data = (
            tau * source_param.data_required + (1 - tau) * target_param.data_required
        )


def hard_update(target: Layer, source: Layer):
    """ターゲットネットワークのハード更新（完全コピー）"""
    for target_param, source_param in zip(target.params(), source.params()):
        target_param.data = source_param.data_required.copy()


# %%
class SACAgent(BaseAgent):
    """SAC（Soft Actor-Critic）エージェント

    最大エントロピー強化学習:
    - 確率方策（ガウス + tanh-squash）
    - Twin Critics（TD3から継承）
    - エントロピー正則化: 目的 = E[Q - α * log π]
    - 自動温度調整（target entropy）
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 256,
        action_scale: float = 2.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        # SAC固有
        init_alpha: float = 0.2,  # 初期温度
        auto_alpha: bool = True,  # 温度の自動調整
        target_entropy: float | None = None,  # 目標エントロピー（Noneで-action_dim）
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps = 0

        # 温度パラメータ（エントロピー係数）
        self.auto_alpha = auto_alpha
        self.log_alpha = np.array([np.log(init_alpha)], dtype=np.float32)
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(action_dim)
        )
        self.alpha_lr = alpha_lr

        # Actor（確率方策）
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim, action_scale)
        # ダミーのforward passで重みを初期化
        dummy_state = Variable(np.zeros((1, state_dim), dtype=np.float32))
        _, _, _, _ = self.actor.sample(dummy_state)
        self.actor_optimizer = Adam(lr=actor_lr).setup(self.actor)

        # Twin Critics（Q1, Q2）- ターゲットはCriticのみ（SACはActor targetなし）
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim)
        # ダミーのforward passで重みを初期化
        dummy_action = Variable(np.zeros((1, action_dim), dtype=np.float32))
        _ = self.critic1.predict(dummy_state, dummy_action)
        _ = self.critic1_target.predict(dummy_state, dummy_action)
        _ = self.critic2.predict(dummy_state, dummy_action)
        _ = self.critic2_target.predict(dummy_state, dummy_action)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)
        self.critic1_optimizer = Adam(lr=critic_lr).setup(self.critic1)
        self.critic2_optimizer = Adam(lr=critic_lr).setup(self.critic2)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, continuous_action=True)

        # 最新のloss（モニタリング用）
        self.last_actor_loss: float | None = None
        self.last_critic1_loss: float | None = None
        self.last_critic2_loss: float | None = None
        self.last_alpha_loss: float | None = None
        self.last_alpha: float = init_alpha
        self.last_entropy: float | None = None

    @property
    def alpha(self) -> float:
        """現在の温度パラメータ"""
        return float(np.exp(self.log_alpha))

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_v = Variable(state.reshape(1, -1).astype(np.float32))

        if explore:
            # 確率方策からサンプル
            action, _, _, _ = self.actor.sample(state_v)
            return action.data_required.flatten().astype(np.float32)
        else:
            # 決定的に平均を使用
            mean, _ = self.actor.predict(state_v)
            action = F.tanh(mean) * self.action_scale
            return action.data_required.flatten().astype(np.float32)

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        terminated=None,
    ):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self) -> dict | None:
        """Replay Bufferからサンプリングして更新"""
        # Warmup期間中は更新しない
        if self.total_steps < self.warmup_steps:
            return None

        # バッファサイズが足りない場合は更新しない
        if len(self.buffer) < self.batch_size:
            return None

        # ミニバッチサンプリング
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_v = Variable(states)
        actions_v = Variable(actions)
        rewards_v = rewards.reshape(-1, 1)
        next_states_v = Variable(next_states)
        dones_v = dones.reshape(-1, 1)

        alpha = self.alpha

        # --- Twin Critics のターゲット計算 ---
        # y = r + γ * (min(Q1', Q2') - α * log π(a'|s'))
        next_actions, next_log_probs, _, _ = self.actor.sample(next_states_v)
        target_q1 = self.critic1_target.predict(
            next_states_v, next_actions
        ).data_required
        target_q2 = self.critic2_target.predict(
            next_states_v, next_actions
        ).data_required
        target_q = np.minimum(target_q1, target_q2)
        # エントロピー項を引く
        next_log_probs_np = next_log_probs.data_required
        td_target = rewards_v + self.gamma * (target_q - alpha * next_log_probs_np) * (
            1 - dones_v
        )
        td_target_v = Variable(td_target.astype(np.float32))

        # --- Critic1 更新 ---
        current_q1 = self.critic1.predict(states_v, actions_v)
        critic1_loss = ((current_q1 - td_target_v) ** 2).sum() / self.batch_size

        self.critic1.cleargrads()
        critic1_loss.backward()
        self.critic1_optimizer.update()

        # --- Critic2 更新 ---
        current_q2 = self.critic2.predict(states_v, actions_v)
        critic2_loss = ((current_q2 - td_target_v) ** 2).sum() / self.batch_size

        self.critic2.cleargrads()
        critic2_loss.backward()
        self.critic2_optimizer.update()

        self.last_critic1_loss = float(critic1_loss.data_required)
        self.last_critic2_loss = float(critic2_loss.data_required)

        # --- Actor 更新 ---
        # L = E[α * log π(a|s) - min(Q1(s,a), Q2(s,a))]
        new_actions, log_probs, _, _ = self.actor.sample(states_v)
        q1_new = self.critic1.predict(states_v, new_actions)
        q2_new = self.critic2.predict(states_v, new_actions)
        # min(Q1, Q2) を使用
        q_new = q1_new  # 簡略化: Q1のみ使用（両方使うと勾配が複雑）
        # Actor損失: α * log π - Q（最小化 = Q最大化 + エントロピー最大化）
        actor_loss = (alpha * log_probs - q_new).sum() / self.batch_size

        self.actor.cleargrads()
        actor_loss.backward()
        self.actor_optimizer.update()

        self.last_actor_loss = float(actor_loss.data_required)
        self.last_entropy = -float(log_probs.data_required.mean())

        # --- 温度(α)の自動調整 ---
        if self.auto_alpha:
            # α損失: -α * (log π + target_entropy)
            # = -α * (log π + H_target)
            # log πが大きい（エントロピー小）→ αを上げたい → 損失を下げる方向
            log_probs_np = log_probs.data_required
            alpha_loss = -self.log_alpha * np.mean(
                log_probs_np + self.target_entropy
            )
            # 勾配降下（手動）
            alpha_grad = -np.mean(log_probs_np + self.target_entropy)
            self.log_alpha -= self.alpha_lr * alpha_grad
            self.last_alpha_loss = float(alpha_loss)

        self.last_alpha = self.alpha

        # --- ターゲットネットワークのソフト更新 ---
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

        return {
            "actor_loss": self.last_actor_loss,
            "critic1_loss": self.last_critic1_loss,
            "critic2_loss": self.last_critic2_loss,
            "alpha": self.last_alpha,
            "entropy": self.last_entropy,
        }

    def start_episode(self):
        pass

    def end_episode(self):
        pass


# %% [markdown]
# ## 統計抽出関数


# %%
def pendulum_stats_extractor(agent) -> str | None:
    """Pendulum Agent用の統計抽出関数"""
    parts = []

    actor_loss = getattr(agent, "last_actor_loss", None)
    critic1_loss = getattr(agent, "last_critic1_loss", None)
    critic2_loss = getattr(agent, "last_critic2_loss", None)
    alpha = getattr(agent, "last_alpha", None)
    entropy = getattr(agent, "last_entropy", None)

    if critic1_loss is not None and critic2_loss is not None:
        critic_loss = (critic1_loss + critic2_loss) / 2
        if actor_loss is not None:
            parts.append(f"A={actor_loss:.3f}, C={critic_loss:.3f}")

    if alpha is not None:
        parts.append(f"α={alpha:.3f}")

    if entropy is not None:
        parts.append(f"H={entropy:.2f}")

    if not parts:
        return None

    return ", " + ", ".join(parts)


def pendulum_eval_stats_extractor(result: EvalResult) -> str:
    """Pendulum評価結果の表示"""
    return f"Return={result.avg_return:.2f}, Mean|Action|={result.mean_abs_action:.3f}"


# %% [markdown]
# ## 実行：SAC

# %%
print("=== Pendulum-v1 SAC (dpl) ===")
print()

# SAC エージェント
agent = SACAgent(
    state_dim=3,
    action_dim=1,
    hidden_dim=256,
    action_scale=2.0,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    buffer_size=100000,
    batch_size=256,
    warmup_steps=1000,
    # SAC固有
    init_alpha=0.2,
    auto_alpha=True,
    target_entropy=-1.0,  # -action_dim
)

# %%
# 環境をラップ
env = GymEnvWrapper(gym.make("Pendulum-v1"))

# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    agent=agent,
    num_episodes=200,
    eval_interval=20,
    eval_n=10,
    update_every=1,  # 毎ステップ更新
    log_interval=20,
    stats_extractor=pendulum_stats_extractor,
    eval_stats_extractor=pendulum_eval_stats_extractor,
)

# %%
result = trainer.train()

# %%
print()
print(f"Total episodes: {len(result.episode_rewards)}")
print(f"Eval checkpoints: {len(result.eval_returns)}")

# %% [markdown]
# ## 学習曲線

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Episode Returns（移動平均付き）
axes[0].plot(result.episode_rewards, alpha=0.3, label="Episode")
window = 20
if len(result.episode_rewards) >= window:
    moving_avg = np.convolve(
        result.episode_rewards, np.ones(window) / window, mode="valid"
    )
    axes[0].plot(
        range(window - 1, len(result.episode_rewards)),
        moving_avg,
        label=f"MA({window})",
    )
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Return")
axes[0].set_title("SAC Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("SAC Evaluation Returns (explore=False)")
    axes[1].grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## エージェントの動作をアニメーション表示

# %%
# フレームを収集
render_env = gym.make("Pendulum-v1", render_mode="rgb_array")
frames = []

obs, _ = render_env.reset()
for _ in range(200):
    frame = render_env.render()
    frames.append(frame)

    action = agent.act(obs, explore=False)
    obs, _, terminated, truncated, _ = render_env.step(action)

    if terminated or truncated:
        break

render_env.close()

# %%
# アニメーション作成・表示
fig, ax = plt.subplots(figsize=(4, 4))
ax.axis("off")
img = ax.imshow(frames[0])


def update(frame_idx):
    img.set_array(frames[frame_idx])
    return [img]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
plt.close(fig)

HTML(anim.to_jshtml())
