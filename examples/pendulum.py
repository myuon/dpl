# %% [markdown]
# # Pendulum-v1 強化学習
#
# A2C（Advantage Actor-Critic）による連続行動空間の学習
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
from dpl.agent_trainer import AgentTrainer, GymEnvWrapper

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
# ## A2C（Advantage Actor-Critic）
#
# TD誤差ベースのActor-Critic。Nステップごとに更新。
#
# - Actor（方策）: π(a|s) = N(μ(s), σ(s))
# - Critic（価値関数）: V(s) → 状態の期待リターンを推定
# - Advantage: A_t = r_t + γV(s_{t+1}) - V(s_t)（TD誤差）
# - Actor損失: -E[log π(a|s) * A_t]
# - Critic損失: MSE(V(s), r + γV(s'))


# %%
from dpl.layers import Layer, Sequential


class GaussianPolicy(Layer):
    """ガウス方策ネットワーク

    状態 s → (μ, σ) を出力
    Layerを継承して params()/cleargrads() を利用
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        action_scale: float = 2.0,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
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

        # log_std をクランプ
        log_std_data = np.clip(
            log_std.data_required, self.log_std_min, self.log_std_max
        )
        log_std = Variable(log_std_data)
        std = F.exp(log_std)

        return mean, std


# %%
class ValueNetwork(Layer):
    """価値関数ネットワーク（Critic）

    状態 s → V(s) を出力
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = Sequential(
            L.Linear(hidden_dim, in_size=state_dim),
            F.relu,
            L.Linear(hidden_dim),
            F.relu,
            L.Linear(1),  # スカラー出力
        )

    def predict(self, state: Variable) -> Variable:
        """状態から価値を出力"""
        return self.net.apply(state)


# %%
class A2CAgent(BaseAgent):
    """A2C（Advantage Actor-Critic）エージェント

    TD誤差ベースのActor-Critic。Nステップごとに更新。
    dpl（自作NNライブラリ）を使用。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 64,
        action_scale: float = 2.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 5,
        critic_update_steps: int = 5,
    ):
        self.gamma = gamma
        self.action_scale = action_scale
        self.n_steps = n_steps
        self.critic_update_steps = critic_update_steps

        # Actor（方策ネットワーク）
        self.policy = GaussianPolicy(
            state_dim, action_dim, hidden_dim, action_scale
        )
        self.actor_optimizer = Adam(lr=actor_lr).setup(self.policy)

        # Critic（価値関数ネットワーク）
        self.critic = ValueNetwork(state_dim, hidden_dim)
        self.critic_optimizer = Adam(lr=critic_lr).setup(self.critic)

        # N-step用バッファ
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []

        # ダミーバッファ（インターフェース互換用）
        self.buffer = ReplayBuffer(capacity=1, continuous_action=True)

        # 最新のloss（モニタリング用）
        self.last_actor_loss: float | None = None
        self.last_critic_loss: float | None = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_v = Variable(state.reshape(1, -1).astype(np.float32))
        mean, std = self.policy.predict(state_v)

        if explore:
            eps = np.random.randn(*mean.shape).astype(np.float32)
            raw_action = mean.data_required + std.data_required * eps
        else:
            raw_action = mean.data_required

        action = self.action_scale * np.tanh(raw_action)
        return action.flatten().astype(np.float32)

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
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        # truncated(時間切れ)ではbootstrapを切らない、terminatedのみ切る
        self.dones.append(terminated if terminated is not None else done)

    def update(self) -> dict | None:
        """Nステップ分たまったら更新"""
        if len(self.states) < self.n_steps:
            return None

        return self._update_from_buffer()

    def _update_from_buffer(self) -> dict:
        """バッファからActor/Criticを更新"""
        states = Variable(np.array(self.states, dtype=np.float32))
        actions = np.array(self.actions, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        next_states = Variable(np.array(self.next_states, dtype=np.float32))
        dones = np.array(self.dones, dtype=np.float32)

        # TD targets: r + γV(s') * (1 - done)
        next_values = self.critic.predict(next_states).data_required.reshape(-1)
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        td_targets_v = Variable(td_targets.reshape(-1, 1).astype(np.float32))

        # 現在の価値
        values = self.critic.predict(states)

        # TD誤差（Advantage）
        advantages = td_targets_v.data_required - values.data_required
        advantages = advantages.reshape(-1)

        # Advantageの正規化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_v = Variable(advantages.astype(np.float32))

        # --- Actor更新 ---
        log_probs = self._compute_log_prob(states, actions)
        actor_loss = -(log_probs * advantages_v).sum() / len(self.states)

        self.policy.cleargrads()
        actor_loss.backward()
        self.actor_optimizer.update()

        # --- Critic更新（複数回） ---
        critic_loss_value = 0.0
        for _ in range(self.critic_update_steps):
            values = self.critic.predict(states)
            critic_loss = ((values - td_targets_v) ** 2).sum() / len(self.states)
            critic_loss_value = float(critic_loss.data_required)

            self.critic.cleargrads()
            critic_loss.backward()
            self.critic_optimizer.update()

        # lossを保存（モニタリング用）
        self.last_actor_loss = float(actor_loss.data_required)
        self.last_critic_loss = critic_loss_value

        # バッファをクリア
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        return {"actor_loss": self.last_actor_loss, "critic_loss": self.last_critic_loss}

    def _compute_log_prob(
        self, states: Variable, actions: np.ndarray
    ) -> Variable:
        """行動に対する log_prob を計算（tanh 補正込み）"""
        mean, std = self.policy.predict(states)

        action_normalized = np.clip(actions / self.action_scale, -0.999, 0.999)
        raw_action = np.arctanh(action_normalized)
        raw_action_v = Variable(raw_action.astype(np.float32))

        diff = raw_action_v - mean
        log_prob = -0.5 * np.log(2 * np.pi) - F.log(std) - 0.5 * (diff / std) ** 2

        tanh_raw = F.tanh(raw_action_v)
        jacobian_correction = F.log(
            Variable(np.full_like(raw_action, self.action_scale, dtype=np.float32))
            * (Variable(np.ones_like(raw_action, dtype=np.float32)) - tanh_raw * tanh_raw)
            + Variable(np.full_like(raw_action, 1e-6, dtype=np.float32))
        )
        log_prob = log_prob - jacobian_correction

        log_prob = log_prob.sum(axis=1)
        return log_prob

    def start_episode(self):
        """エピソード開始時にバッファをクリア"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def end_episode(self):
        """エピソード終了時に残りのバッファを更新"""
        if len(self.states) > 0:
            self._update_from_buffer()


# %% [markdown]
# ## 統計抽出関数

# %%
def pendulum_stats_extractor(agent) -> str | None:
    """Pendulum Agent用の統計抽出関数"""
    parts = []

    actor_loss = getattr(agent, "last_actor_loss", None)
    critic_loss = getattr(agent, "last_critic_loss", None)
    if actor_loss is not None and critic_loss is not None:
        parts.append(f"ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss:.4f}")

    if not parts:
        return None

    return ", " + ", ".join(parts)


# %% [markdown]
# ## 実行：A2C

# %%
print("=== Pendulum-v1 A2C (dpl) ===")
print()

# A2C エージェント
agent = A2CAgent(
    state_dim=3,
    action_dim=1,
    hidden_dim=64,
    action_scale=2.0,
    actor_lr=3e-4,
    critic_lr=1e-3,
    gamma=0.99,
    n_steps=5,
    critic_update_steps=5,
)

# %%
# 環境をラップ
env = GymEnvWrapper(gym.make("Pendulum-v1"))
eval_env = GymEnvWrapper(gym.make("Pendulum-v1"))

# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    eval_env=eval_env,
    agent=agent,
    num_episodes=500,
    eval_interval=50,
    eval_n=10,
    update_every=1,
    log_interval=50,
    stats_extractor=pendulum_stats_extractor,
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
        range(window - 1, len(result.episode_rewards)), moving_avg, label=f"MA({window})"
    )
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Return")
axes[0].set_title("Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("Evaluation Returns (explore=False)")
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
