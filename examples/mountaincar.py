# %% [markdown]
# # MountainCarContinuous-v0 強化学習
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
# | 観測 | shape=(2,), [position, velocity] |
# | 観測範囲 | [-1.2, -0.07] ~ [0.6, 0.07] |
# | 行動 | shape=(1,), 連続値力 |
# | 行動範囲 | [-1, 1] |
# | 報酬 | -0.1 * action² + 100（position >= 0.45 でゴール） |
# | terminated | position >= 0.45 でTrue |
# | truncated | 999ステップでTrue |
#
# ## 難しさ
# - **スパース報酬**: ゴール到達時の+100のみ意味がある
# - **探索が重要**: 右に登るには左に振ってから右へ（反直感的）
# - **長いエピソード**: 最大999ステップ

# %% [markdown]
# ## SAC（Soft Actor-Critic）
#
# 最大エントロピー強化学習。探索を"ノイズで足す"から"目的関数で制御する"へ。
# スパース報酬環境でもエントロピー正則化により探索を維持。
#
# ## 更新式
# - Target: y = r + γ * (min(Q1'(s',a'), Q2'(s',a')) - α * log π(a'|s'))
# - Critic: L = E[(Q(s,a) - y)²]
# - Actor: L = E[α * log π(a|s) - Q(s, a)]
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
        action_scale: float = 1.0,
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
        log_prob = self._compute_log_prob(z, mean, log_std)

        return action, log_prob, mean, log_std

    def _compute_log_prob(
        self, z: Variable, mean: Variable, log_std: Variable
    ) -> Variable:
        """log π(a|s) を計算（tanh補正込み）"""
        std = F.exp(log_std)

        # ガウス分布のlog確率
        diff = z - mean
        log_prob = (
            -0.5 * np.log(2 * np.pi)
            - log_std
            - 0.5 * (diff / std) ** 2
        )

        # tanh補正: -log(1 - tanh²(z))
        tanh_z = F.tanh(z)
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
        self.fc1 = L.Linear(hidden_dim, in_size=state_dim + action_dim)
        self.fc2 = L.Linear(hidden_dim)
        self.fc3 = L.Linear(1)

    def predict(self, state: Variable, action: Variable) -> Variable:
        """状態と行動からQ値を出力"""
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
    - Twin Critics
    - エントロピー正則化: 目的 = E[Q - α * log π]
    - 自動温度調整（target entropy）
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 256,
        action_scale: float = 1.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        # SAC固有
        init_alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
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
        dummy_state = Variable(np.zeros((1, state_dim), dtype=np.float32))
        _, _, _, _ = self.actor.sample(dummy_state)
        self.actor_optimizer = Adam(lr=actor_lr).setup(self.actor)

        # Twin Critics
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim)
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
        return float(np.exp(self.log_alpha))

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_v = Variable(state.reshape(1, -1).astype(np.float32))

        if explore:
            action, _, _, _ = self.actor.sample(state_v)
            return action.data_required.flatten().astype(np.float32)
        else:
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
        if self.total_steps < self.warmup_steps:
            return None

        if len(self.buffer) < self.batch_size:
            return None

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
        next_actions, next_log_probs, _, _ = self.actor.sample(next_states_v)
        target_q1 = self.critic1_target.predict(
            next_states_v, next_actions
        ).data_required
        target_q2 = self.critic2_target.predict(
            next_states_v, next_actions
        ).data_required
        target_q = np.minimum(target_q1, target_q2)
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
        new_actions, log_probs, _, _ = self.actor.sample(states_v)
        q1_new = self.critic1.predict(states_v, new_actions)
        actor_loss = (alpha * log_probs - q1_new).sum() / self.batch_size

        self.actor.cleargrads()
        actor_loss.backward()
        self.actor_optimizer.update()

        self.last_actor_loss = float(actor_loss.data_required)
        self.last_entropy = -float(log_probs.data_required.mean())

        # --- 温度(α)の自動調整 ---
        if self.auto_alpha:
            log_probs_np = log_probs.data_required
            alpha_loss = -self.log_alpha * np.mean(
                log_probs_np + self.target_entropy
            )
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
def mountaincar_stats_extractor(agent) -> str | None:
    """MountainCar Agent用の統計抽出関数"""
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


def mountaincar_eval_stats_extractor(result: EvalResult) -> str:
    """MountainCar評価結果の表示"""
    return (
        f"Return={result.avg_return:.2f}, "
        f"Success={result.success_rate*100:.0f}%, "
        f"AvgSteps={result.avg_steps:.1f}"
    )


# %% [markdown]
# ## 実行：SAC

# %%
print("=== MountainCarContinuous-v0 SAC (dpl) ===")
print()

# SAC エージェント
# スパース報酬対策：
# - 高い初期α（探索促進）
# - 大きめのbuffer（長いエピソードを蓄積）
# - warmup_stepsを長め（探索データを貯める）
agent = SACAgent(
    state_dim=2,
    action_dim=1,
    hidden_dim=256,
    action_scale=1.0,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    buffer_size=100000,
    batch_size=256,
    warmup_steps=1000,
    # SAC固有
    init_alpha=0.5,  # 高めの初期温度（探索促進）
    auto_alpha=True,
    target_entropy=-1.0,
)

# %%
# 環境をラップ
env = GymEnvWrapper(gym.make("MountainCarContinuous-v0"))

# AgentTrainerで学習
# スパース報酬なので長めに学習
trainer = AgentTrainer(
    env=env,
    agent=agent,
    num_episodes=500,
    eval_interval=50,
    eval_n=10,
    update_every=1,
    log_interval=50,
    stats_extractor=mountaincar_stats_extractor,
    eval_stats_extractor=mountaincar_eval_stats_extractor,
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
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Episode Returns（移動平均付き）
axes[0].plot(result.episode_rewards, alpha=0.3, label="Episode")
window = 50
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
    axes[1].set_title("SAC Evaluation Returns")
    axes[1].grid(True)

# Success Rate
if result.eval_success_rates:
    episodes, rates = zip(*result.eval_success_rates)
    axes[2].plot(episodes, [r * 100 for r in rates], marker="o", color="green")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_title("Goal Achievement Rate")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## エージェントの動作をアニメーション表示

# %%
# フレームを収集
render_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
frames = []

obs, _ = render_env.reset()
total_reward = 0
for step in range(999):
    frame = render_env.render()
    frames.append(frame)

    action = agent.act(obs, explore=False)
    obs, reward, terminated, truncated, _ = render_env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

render_env.close()
print(f"Evaluation: steps={step+1}, reward={total_reward:.2f}, goal={'Yes' if terminated else 'No'}")

# %%
# アニメーション作成・表示
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")
img = ax.imshow(frames[0])


def update(frame_idx):
    img.set_array(frames[frame_idx])
    return [img]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
plt.close(fig)

HTML(anim.to_jshtml())
