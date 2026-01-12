# %% [markdown]
# # MiniGrid DQN
#
# DQNを用いたMiniGrid環境の学習
# Partial Observation (エージェント視点 7x7) モードを使用

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

import minigrid
from gymnasium import ActionWrapper

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.agent_trainer import AgentTrainer, EvalResult
from models.dqn import DQNAgent, dqn_stats_extractor

# %% [markdown]
# ## 環境仕様
#
# | 項目 | 内容 |
# |------|------|
# | 環境 | Train: Empty-6x6, Eval: Empty-Random-6x6 |
# | 観測 | shape=(7, 7, 3), 0-10のuint8 (Partial Obs: エージェント視点) |
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
    """MiniGrid用のMLP Q-Network

    フラット化した観測を入力とするシンプルなMLP
    ※ MiniGrid観測はカテゴリ値 (object_id, color_id, state_id) なので
      まずMLPで動作確認してからCNN化を検討
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # シンプルなMLP
        self.net = L.Sequential(
            L.Linear(hidden_size, in_size=state_size),
            F.relu,
            L.Linear(hidden_size),
            F.relu,
            L.Linear(action_size),
        )

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs
        return self.net(x)


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

    Partial Observation (エージェント視点 7x7) を使用
    image + direction を連結して返す
    """

    def __init__(self, env_name: str = "MiniGrid-Empty-8x8-v0"):
        self.base_env = gym.make(env_name)
        self.env = SimpleActionWrapper(self.base_env)
        # image shape + direction (4方向 one-hot)
        img_shape = self.env.observation_space["image"].shape
        assert img_shape is not None
        self.obs_shape = img_shape
        self.state_size = int(np.prod(img_shape)) + 4  # image + direction one-hot

    def _process_obs(self, obs: dict) -> np.ndarray:
        """image + direction one-hot をフラット化"""
        image = obs["image"].flatten().astype(np.float32) / 10.0
        direction = obs["direction"]
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[direction] = 1.0
        return np.concatenate([image, direction_onehot])

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        return self._process_obs(obs)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
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
print("=== MiniGrid DQN ===")
print("Train: MiniGrid-Empty-6x6-v0")
print("Eval:  MiniGrid-Empty-Random-6x6-v0")
print()

# 環境セットアップ
env = MiniGridEnvWrapper("MiniGrid-Empty-6x6-v0")
eval_env = MiniGridEnvWrapper("MiniGrid-Empty-Random-6x6-v0")
print(f"Observation shape: {env.obs_shape}")
print()

# Q-Network作成
state_size = env.state_size
action_size = 3  # left, right, forward（Empty環境用）
hidden_size = 128

qnet = QNet(state_size, action_size, hidden_size)
target_qnet = QNet(state_size, action_size, hidden_size)

# ダミー入力で重みを初期化
dummy_input = Variable(np.zeros((1, state_size), dtype=np.float32))
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

# フレームを収集 (eval環境で動作確認)
render_env = SimpleActionWrapper(
    gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode="rgb_array")
)
frames = []
action_names = ["left", "right", "forward"]
action_history = []


def process_obs_for_render(obs: dict) -> np.ndarray:
    """観測をエージェント用にフラット化"""
    image = obs["image"].flatten().astype(np.float32) / 10.0
    direction = obs["direction"]
    direction_onehot = np.zeros(4, dtype=np.float32)
    direction_onehot[direction] = 1.0
    return np.concatenate([image, direction_onehot])


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
