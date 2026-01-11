# %% [markdown]
# # Pendulum-v1 強化学習の土台
#
# env reset → rollout → buffer/store → update → eval の骨組み
# 手法は後から差し替え可能な設計

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

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
# ## Random Agent（動作確認用）


# %%
class RandomAgent(BaseAgent):
    """ランダム行動エージェント（骨組み確認用）"""

    def __init__(self, action_low: float = -2.0, action_high: float = 2.0):
        self.action_low = action_low
        self.action_high = action_high
        self.buffer = ReplayBuffer(capacity=100000, continuous_action=True)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        return np.random.uniform(self.action_low, self.action_high, size=(1,)).astype(
            np.float32
        )

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

    def update(self) -> dict | None:
        return None  # 学習なし


# %% [markdown]
# ## ガウスノイズ付き定数トルク（ベースライン）
#
# ランダムよりマシな制御のベースライン。
# 行動のスケーリング（クリッピング）の確認用。


# %%
class NoisyConstantAgent(BaseAgent):
    """ガウスノイズ付き定数トルクエージェント

    base_torque にガウスノイズを加えた行動を出力。
    tanh でスケーリング: action = action_scale * tanh(raw)
    """

    def __init__(
        self,
        base_torque: float = 0.0,
        noise_sigma: float = 1.0,
        action_scale: float = 2.0,
    ):
        self.base_torque = base_torque
        self.noise_sigma = noise_sigma
        self.action_scale = action_scale
        self.buffer = ReplayBuffer(capacity=100000, continuous_action=True)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        # obs = [cos(θ), sin(θ), θ_dot]
        # θ=0 が真上（目標位置）
        theta = np.arctan2(state[1], state[0])

        # raw は unbounded (-inf, inf)
        # θ > 0 なら負のトルク、θ < 0 なら正のトルク（P制御的）
        raw = -theta

        if explore:
            raw = raw + np.random.normal(0, self.noise_sigma)

        # tanh でスケーリング
        a = np.tanh(raw)  # [-1, 1]
        action = self.action_scale * a  # [-action_scale, action_scale]

        return np.array([action], dtype=np.float32)

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

    def update(self) -> dict | None:
        return None  # 学習なし


# %% [markdown]
# ## ガウスサンプリング（方策ネット出力のシミュレーション）
#
# 方策ネットワークの出力を模擬：
# raw ~ N(mu, sigma), action = action_scale * tanh(raw)


# %%
class GaussianSamplingAgent(BaseAgent):
    """ガウス分布からサンプルして tanh スケーリング

    方策ネットワークの出力を模擬するテスト用。
    raw ~ N(mu, sigma)
    action = action_scale * tanh(raw)
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        action_scale: float = 2.0,
    ):
        self.mu = mu
        self.sigma = sigma
        self.action_scale = action_scale
        self.buffer = ReplayBuffer(capacity=100000, continuous_action=True)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        if explore:
            raw = np.random.normal(self.mu, self.sigma)
        else:
            raw = self.mu

        # tanh でスケーリング
        action = self.action_scale * np.tanh(raw)

        return np.array([action], dtype=np.float32)

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

    def update(self) -> dict | None:
        return None  # 学習なし


# %% [markdown]
# ## 実行：骨組み確認

# %%
print("=== Pendulum-v1 ベースライン確認 ===")
print()

# ガウスノイズ付き定数トルクエージェント
# base_torque=0, sigma=1.0: ほぼランダムだが中心が0
agent = NoisyConstantAgent(base_torque=0.0, noise_sigma=1.0, action_scale=2.0)

# %%
# 環境をラップ
env = GymEnvWrapper(gym.make("Pendulum-v1"))
eval_env = GymEnvWrapper(gym.make("Pendulum-v1"))

# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    eval_env=eval_env,
    agent=agent,
    num_episodes=50,
    eval_interval=10,
    eval_n=10,
    update_every=1,
    log_interval=10,
)

# %%
result = trainer.train()

# %%
print()
print(f"Total episodes: {len(result.episode_rewards)}")
print(f"Eval checkpoints: {len(result.eval_returns)}")
print(f"Buffer size: {len(agent.buffer)}")

# %% [markdown]
# ## 学習曲線

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Episode Returns
axes[0].plot(result.episode_rewards)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Return")
axes[0].set_title("Episode Returns")
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("Evaluation Returns")
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

    # explore=True でノイズ込みの行動を確認
    action = agent.act(obs, explore=True)
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
