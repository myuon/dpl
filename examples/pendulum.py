# %% [markdown]
# # Pendulum-v1 強化学習
#
# TD3（Twin Delayed Deep Deterministic Policy Gradient）による連続行動空間の学習
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
# ## TD3（Twin Delayed Deep Deterministic Policy Gradient）
#
# DDPGの3つの問題を解決した改良版。SACの前段階として重要。
#
# ## TD3の特徴（DDPGからの改良点）
# 1. **Twin Critics（Double Q）**: Q1, Q2 の2つを学習し、ターゲットは min(Q1, Q2) を使用
#    → Q値の過大評価を防ぐ
# 2. **Delayed Policy Updates**: Criticを複数回更新するごとにActorを1回更新
#    → Criticが十分学習してからActorを更新
# 3. **Target Policy Smoothing**: ターゲット方策の行動にノイズを加える
#    → Q関数の過学習を防ぐ
#
# ## 更新式
# - Target: y = r + γ * min(Q1'(s', μ'(s') + ε), Q2'(s', μ'(s') + ε))
# - Critic: L = E[(Q1(s,a) - y)²] + E[(Q2(s,a) - y)²]
# - Actor: ∇_θ E[Q1(s, μ_θ(s))] を最大化（policy_delay回に1回）


# %%
from dpl.layers import Layer, Sequential


class DeterministicPolicy(Layer):
    """決定的方策ネットワーク（TD3 Actor）

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
    """Q関数ネットワーク（TD3 Critic）

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
class TD3Agent(BaseAgent):
    """TD3（Twin Delayed Deep Deterministic Policy Gradient）エージェント

    DDPGの3つの問題を解決:
    1. Twin Critics: Q1, Q2 → min(Q1, Q2) でターゲット計算（過大評価防止）
    2. Delayed Policy Updates: Criticを複数回更新後にActorを1回更新
    3. Target Policy Smoothing: ターゲット行動にノイズを加える
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
        # TD3固有のハイパーパラメータ
        policy_delay: int = 2,  # Actorの更新頻度（Critic更新N回に1回）
        target_noise: float = 0.2,  # ターゲット行動へのノイズ
        noise_clip: float = 0.5,  # ターゲットノイズのクリップ範囲
        warmup_steps: int = 1000,
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.update_count = 0  # Critic更新回数（policy_delay判定用）

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

        # Twin Critics（Q1, Q2）
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim)
        # ダミーのforward passで重みを初期化
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

        # --- Target Policy Smoothing ---
        # ターゲット行動にノイズを加える（Q関数の過学習防止）
        next_actions = self.actor_target.predict(next_states_v).data_required
        noise = np.clip(
            np.random.randn(*next_actions.shape) * self.target_noise,
            -self.noise_clip,
            self.noise_clip,
        )
        next_actions_noisy = np.clip(
            next_actions + noise, -self.action_scale, self.action_scale
        )
        next_actions_v = Variable(next_actions_noisy.astype(np.float32))

        # --- Twin Critics のターゲット計算（Double Q） ---
        # min(Q1', Q2') で過大評価を防ぐ
        target_q1 = self.critic1_target.predict(
            next_states_v, next_actions_v
        ).data_required
        target_q2 = self.critic2_target.predict(
            next_states_v, next_actions_v
        ).data_required
        target_q = np.minimum(target_q1, target_q2)  # Clipped Double Q
        td_target = rewards_v + self.gamma * target_q * (1 - dones_v)
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

        self.update_count += 1

        # --- Delayed Policy Updates ---
        # policy_delay回に1回だけActorとターゲットを更新
        if self.update_count % self.policy_delay == 0:
            # Actor更新: Q1のみ使用（論文通り）
            predicted_actions = self.actor.predict(states_v)
            actor_loss = (
                -self.critic1.predict(states_v, predicted_actions).sum()
                / self.batch_size
            )

            self.actor.cleargrads()
            actor_loss.backward()
            self.actor_optimizer.update()

            self.last_actor_loss = float(actor_loss.data_required)

            # ターゲットネットワークのソフト更新
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

        # Noneを含まない辞書を返す
        result = {
            "critic1_loss": self.last_critic1_loss,
            "critic2_loss": self.last_critic2_loss,
        }
        if self.last_actor_loss is not None:
            result["actor_loss"] = self.last_actor_loss
        return result

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

    if critic1_loss is not None and critic2_loss is not None:
        critic_loss = (critic1_loss + critic2_loss) / 2
        if actor_loss is not None:
            parts.append(f"ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss:.4f}")
        else:
            parts.append(f"CriticLoss={critic_loss:.4f}")

    if not parts:
        return None

    return ", " + ", ".join(parts)


def pendulum_eval_stats_extractor(result: EvalResult) -> str:
    """Pendulum評価結果の表示"""
    return f"Return={result.avg_return:.2f}, Mean|Action|={result.mean_abs_action:.3f}"


# %% [markdown]
# ## 実行：TD3

# %%
print("=== Pendulum-v1 TD3 (dpl) ===")
print()

# TD3 エージェント
agent = TD3Agent(
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
    # TD3固有
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
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
axes[0].set_title("TD3 Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("TD3 Evaluation Returns (explore=False)")
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
