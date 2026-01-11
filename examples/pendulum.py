# %% [markdown]
# # Pendulum-v1 強化学習
#
# DDPG（Deep Deterministic Policy Gradient）による連続行動空間の学習
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
# ## DDPG（Deep Deterministic Policy Gradient）
#
# オフポリシーのActor-Critic。SACの前段階として重要。
#
# ## DDPGの特徴
# - **Replay Buffer**: 経験を貯めてミニバッチ学習（サンプル効率向上）
# - **Target Network**: θ' ← τθ + (1-τ)θ'（学習の安定化）
# - **Deterministic Policy**: π(s) = μ(s)（確定的方策）
# - **Q学習**: Q(s,a) を学習（状態と行動の両方を入力）
#
# ## 更新式
# - Critic: L = E[(Q(s,a) - (r + γQ'(s', μ'(s'))))²]
# - Actor: ∇_θ E[Q(s, μ_θ(s))] を最大化
#
# ## 探索
# - 探索ノイズ: a = μ(s) + ε（ガウスノイズ）


# %%
from dpl.layers import Layer, Sequential


class DeterministicPolicy(Layer):
    """決定的方策ネットワーク（DDPG Actor）

    状態 s → action を直接出力（tanh でスケーリング）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 2.0,
    ):
        super().__init__()
        self.action_scale = action_scale

        self.net = Sequential(
            L.Linear(hidden_dim, in_size=state_dim),
            F.relu,
            L.Linear(hidden_dim),
            F.relu,
            L.Linear(action_dim),
            F.tanh,
        )

    def predict(self, state: Variable) -> Variable:
        """状態から行動を出力（-action_scale ~ action_scale）"""
        return self.net.apply(state) * self.action_scale


# %%
class QNetwork(Layer):
    """Q関数ネットワーク（DDPG Critic）

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
class DDPGAgent(BaseAgent):
    """DDPG（Deep Deterministic Policy Gradient）エージェント

    オフポリシーのActor-Critic。
    - Replay Buffer でサンプル効率向上
    - Target Network で学習安定化
    - 探索ノイズで探索
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 256,
        action_scale: float = 2.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        warmup_steps: int = 1000,
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.warmup_steps = warmup_steps
        self.total_steps = 0

        # Actor（決定的方策）
        self.actor = DeterministicPolicy(
            state_dim, action_dim, hidden_dim, action_scale
        )
        self.actor_target = DeterministicPolicy(
            state_dim, action_dim, hidden_dim, action_scale
        )
        # ダミーのforward passで重みを初期化
        dummy_state = Variable(np.zeros((1, state_dim), dtype=np.float32))
        dummy_action = self.actor.predict(dummy_state)
        _ = self.actor_target.predict(dummy_state)
        hard_update(self.actor_target, self.actor)
        self.actor_optimizer = Adam(lr=actor_lr).setup(self.actor)

        # Critic（Q関数）
        self.critic = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim)
        # ダミーのforward passで重みを初期化
        _ = self.critic.predict(dummy_state, dummy_action)
        _ = self.critic_target.predict(dummy_state, dummy_action)
        hard_update(self.critic_target, self.critic)
        self.critic_optimizer = Adam(lr=critic_lr).setup(self.critic)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, continuous_action=True)

        # 最新のloss（モニタリング用）
        self.last_actor_loss: float | None = None
        self.last_critic_loss: float | None = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_v = Variable(state.reshape(1, -1).astype(np.float32))
        action = self.actor.predict(state_v).data_required.flatten()

        if explore:
            # ガウスノイズで探索
            noise = (
                np.random.randn(*action.shape)
                * self.exploration_noise
                * self.action_scale
            )
            action = action + noise
            action = np.clip(action, -self.action_scale, self.action_scale)

        return action.astype(np.float32)

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

        # --- Critic更新 ---
        # TDターゲット: y = r + γ * Q'(s', μ'(s'))
        next_actions = self.actor_target.predict(next_states_v)
        target_q = self.critic_target.predict(next_states_v, next_actions).data_required
        td_target = rewards_v + self.gamma * target_q * (1 - dones_v)
        td_target_v = Variable(td_target.astype(np.float32))

        # 現在のQ値
        current_q = self.critic.predict(states_v, actions_v)

        # Critic損失: MSE
        critic_loss = ((current_q - td_target_v) ** 2).sum() / self.batch_size

        self.critic.cleargrads()
        critic_loss.backward()
        self.critic_optimizer.update()

        # --- Actor更新 ---
        # Actor損失: -E[Q(s, μ(s))]（Q値を最大化）
        predicted_actions = self.actor.predict(states_v)
        actor_loss = (
            -self.critic.predict(states_v, predicted_actions).sum() / self.batch_size
        )

        self.actor.cleargrads()
        actor_loss.backward()
        self.actor_optimizer.update()

        # --- ターゲットネットワークのソフト更新 ---
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        # lossを保存
        self.last_actor_loss = float(actor_loss.data_required)
        self.last_critic_loss = float(critic_loss.data_required)

        return {
            "actor_loss": self.last_actor_loss,
            "critic_loss": self.last_critic_loss,
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
    critic_loss = getattr(agent, "last_critic_loss", None)
    if actor_loss is not None and critic_loss is not None:
        parts.append(f"ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss:.4f}")

    if not parts:
        return None

    return ", " + ", ".join(parts)


def pendulum_eval_stats_extractor(result: EvalResult) -> str:
    """Pendulum評価結果の表示"""
    return f"Return={result.avg_return:.2f}, Mean|Action|={result.mean_abs_action:.3f}"


# %% [markdown]
# ## 実行：DDPG

# %%
print("=== Pendulum-v1 DDPG (dpl) ===")
print()

# DDPG エージェント
agent = DDPGAgent(
    state_dim=3,
    action_dim=1,
    hidden_dim=256,
    action_scale=2.0,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    buffer_size=100000,
    batch_size=256,
    exploration_noise=0.1,
    warmup_steps=1000,
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
axes[0].set_title("DDPG Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("DDPG Evaluation Returns (explore=False)")
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
