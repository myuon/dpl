# %% [markdown]
# # MiniGrid DQN
#
# DQNを用いたMiniGrid環境の学習

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

import minigrid
from minigrid.wrappers import FullyObsWrapper
from gymnasium import ActionWrapper

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.core import as_jax  # JAX配列への変換用
from dpl.agent_trainer import AgentTrainer, EvalResult
from models.dqn import DQNAgent, dqn_stats_extractor

# %% [markdown]
# ## 環境仕様
#
# | 項目 | 内容 |
# |------|------|
# | 環境 | ENV_NAME で指定 |
# | 観測 | shape=(C, H, W), 0-10のuint8を正規化 |
# | 行動 | Discrete(3): left, right, forward（Empty環境用に制限）|
# | 報酬 | ゴール到達時に 1 - 0.9 * (step_count / max_steps) |
# | terminated | ゴール到達時にTrue |
# | truncated | max_steps到達時にTrue |
#
# ## 観測エンコーディング
# 各セルは3つの値 (object_type, color, state) を持つ:
# - object_type: 0=unseen, 1=empty, 2=wall, 3=floor, 4=door, 5=key, 6=ball, 7=box, 8=goal, 9=lava, 10=agent
# - color: 0-5 (red, green, blue, purple, yellow, grey)
# - state: 0-2 (open, closed, locked for doors)


# %%
class QNet(L.Layer):
    """MiniGrid用のCNN Q-Network

    入力: (N, C, H, W) 形式の画像
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],  # (H, W, C)
        action_size: int = 3,
    ):
        super().__init__()
        H, W, C = obs_shape
        self.obs_shape = obs_shape
        self.action_size = action_size

        # CNN特徴抽出
        self.conv = L.Sequential(
            L.Conv2d(16, kernel_size=3, stride=1, pad=1, in_channels=C),
            F.relu,
            L.Conv2d(32, kernel_size=3, stride=1, pad=1),
            F.relu,
        )

        # CNN出力サイズを計算 (padding=1, stride=1 なので H, W は変わらない)
        conv_out_size = 32 * H * W

        # 全結合層
        self.fc = L.Sequential(
            L.Linear(128, in_size=conv_out_size),
            F.relu,
            L.Linear(action_size),
        )

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs
        # CNN
        h = self.conv(x)
        # Flatten
        h = F.reshape(h, (h.shape[0], -1))
        # FC
        return self.fc(h)


# %% [markdown]
# ## MiniGrid用ラッパー


# %%
class SimpleActionWrapper(ActionWrapper):
    """Empty環境用にアクション空間を3に制限するラッパー

    MiniGridは7アクション (left, right, forward, pickup, drop, toggle, done) だが
    Empty環境では left, right, forward のみ使用
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        # 0:left, 1:right, 2:forward をそのまま渡す
        return action


class MiniGridEnvWrapper:
    """MiniGrid用のラッパー

    image + direction を連結して返す
    """

    def __init__(self, env_name: str = "MiniGrid-Empty-8x8-v0", fully_obs: bool = True):
        self.base_env = gym.make(env_name)
        if fully_obs:
            self.env = SimpleActionWrapper(FullyObsWrapper(self.base_env))
        else:
            self.env = SimpleActionWrapper(self.base_env)
        # image shape (H, W, C)
        img_shape = self.env.observation_space["image"].shape
        assert img_shape is not None
        self.obs_shape = img_shape  # (H, W, C)

    def _process_obs(self, obs: dict):
        """image を (C, H, W) 形式に変換し、JAX配列としてGPU計算を有効化"""
        # (H, W, C) -> (C, H, W)
        image = obs["image"].transpose(2, 0, 1).astype(np.float32) / 10.0
        return as_jax(image)

    def reset(self):
        obs, info = self.env.reset()
        return self._process_obs(obs)

    def step(self, action: int | np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._process_obs(obs),
            float(reward),
            terminated,
            truncated,
            info,
        )


# %% [markdown]
# ## 統計抽出関数


# %%
def minigrid_eval_stats_extractor(result: EvalResult) -> str:
    """MiniGrid評価結果の表示"""
    return (
        f"Return={result.avg_return:.3f}, "
        f"Success={result.success_rate*100:.0f}%, "
        f"AvgSteps={result.avg_steps:.1f}"
    )


# %% [markdown]
# ## 実行：DQN

# %%
ENV_NAME = "MiniGrid-SimpleCrossingS9N1-v0"
FULLY_OBS = True

print("=== MiniGrid DQN ===")
print(f"Env: {ENV_NAME}")
print(f"Obs: {'FullyObs' if FULLY_OBS else 'PartialObs'}")
print()

# 環境セットアップ
env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
eval_env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
print(f"Observation shape: {env.obs_shape}")
print()

# Q-Network作成
obs_shape = env.obs_shape  # (H, W, C)
action_size = 3  # left, right, forward

qnet = QNet(obs_shape, action_size)
target_qnet = QNet(obs_shape, action_size)

# ダミー入力で重みを初期化 (N, C, H, W) - JAX配列でGPU計算を有効化
H, W, C = obs_shape
dummy_input = Variable(as_jax(np.zeros((1, C, H, W), dtype=np.float32)))
qnet(dummy_input)
target_qnet(dummy_input)

# DQN エージェント
agent = DQNAgent(
    qnet=qnet,
    target_qnet=target_qnet,
    action_size=action_size,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    lr=1e-3,
    batch_size=64,
    buffer_size=10000,
    tau=0.005,
    warmup_steps=500,
)

# %%
# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    eval_env=eval_env,
    agent=agent,
    num_episodes=300,
    eval_interval=30,
    eval_n=10,
    update_every=1,
    log_interval=30,
    stats_extractor=dqn_stats_extractor,
    eval_stats_extractor=minigrid_eval_stats_extractor,
)

# %%
result = trainer.train()

# %%
print()
print(f"Total episodes: {len(result.episode_rewards)}")
print(f"Final epsilon: {agent.epsilon:.4f}")

# %% [markdown]
# ## 学習曲線

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Episode Returns
axes[0].plot(result.episode_rewards, alpha=0.3, label="Episode")
window = 30
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
axes[0].set_title("DQN Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("Evaluation Returns")
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
from matplotlib import animation
from IPython.display import HTML

# フレームを収集
if FULLY_OBS:
    render_env = SimpleActionWrapper(
        FullyObsWrapper(gym.make(ENV_NAME, render_mode="rgb_array"))
    )
else:
    render_env = SimpleActionWrapper(
        gym.make(ENV_NAME, render_mode="rgb_array")
    )
frames = []
action_names = ["left", "right", "forward"]
action_history = []


def process_obs_for_render(obs: dict) -> np.ndarray:
    """観測を (C, H, W) 形式に変換"""
    image = obs["image"].transpose(2, 0, 1).astype(np.float32) / 10.0
    return image


obs, _ = render_env.reset()
obs_flat = process_obs_for_render(obs)

for step in range(50):
    # rgb_array でレンダリング
    frame = render_env.render()
    frames.append(frame)

    action = agent.act(obs_flat, explore=False)
    action_history.append(action_names[action])
    obs, reward, terminated, truncated, _ = render_env.step(action)
    obs_flat = process_obs_for_render(obs)

    if terminated or truncated:
        # 最後のフレームも追加
        frames.append(render_env.render())
        break

render_env.close()
print(f"Episode: {step+1} steps, terminated={terminated}")
print(
    f"Actions: {' -> '.join(action_history[:15])}{'...' if len(action_history) > 15 else ''}"
)

# %%
# アニメーション作成・表示
fig, ax = plt.subplots(figsize=(5, 5))
ax.axis("off")
img = ax.imshow(frames[0])


def update(frame_idx):
    img.set_array(frames[frame_idx])
    return [img]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
plt.close(fig)

HTML(anim.to_jshtml())
