# %% [markdown]
# # MiniGrid DRQN
#
# DRQN (DQN + RNN) を用いた部分観測MiniGrid環境の学習
#
# - 部分観測モード（FULLY_OBS=False）でRNNの効果を確認
# - TimeLSTMによる時系列処理
# - EpisodeReplayによるシーケンスサンプリング

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import minigrid
from minigrid.wrappers import FullyObsWrapper
from gymnasium import ActionWrapper

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.agent_trainer import AgentTrainer, EvalResult
from models.drqn import DRQNNet, DRQNAgent, drqn_stats_extractor

# %% [markdown]
# ## 環境仕様
#
# | 項目 | 内容 |
# |------|------|
# | 環境 | MiniGrid-Empty-16x16-v0 |
# | 観測 | **部分観測**: shape=(7, 7, 3), エージェント視点の7x7グリッド |
# | 行動 | Discrete(3): left, right, forward |
# | 報酬 | ゴール到達時に 1 - 0.9 * (step_count / max_steps) |
#
# ## DRQN のポイント
# - obs_t → MLP → feature_t → LSTM (with h_{t-1}) → h_t → Q(s_t, a)
# - ランダム行動でもLSTM状態を更新（get_action内でstep()呼び出し）
# - burn-in: 最初のNステップはloss計算から除外


# %%
class SimpleActionWrapper(ActionWrapper):
    """Empty環境用にアクション空間を3に制限するラッパー"""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action


class MiniGridEnvWrapper:
    """MiniGrid用のラッパー

    FULLY_OBS=False時は部分観測（7x7グリッド）
    """

    def __init__(self, env_name: str = "MiniGrid-Empty-8x8-v0", fully_obs: bool = True):
        self.base_env = gym.make(env_name)
        if fully_obs:
            self.env = SimpleActionWrapper(FullyObsWrapper(self.base_env))
        else:
            # 部分観測モード（デフォルトの7x7視野）
            self.env = SimpleActionWrapper(self.base_env)
        # image shape (H, W, C)
        img_shape = self.env.observation_space["image"].shape
        assert img_shape is not None
        self.obs_shape = img_shape  # (H, W, C)
        self.state_size = int(np.prod(img_shape))  # フラット化後のサイズ

    def _process_obs(self, obs: dict) -> np.ndarray:
        """image をフラット化"""
        image = obs["image"].flatten().astype(np.float32) / 10.0
        return image

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
# ## 実行：DRQN（部分観測）

# %%
ENV_NAME = "MiniGrid-Empty-8x8-v0"
FULLY_OBS = False  # 部分観測モード

print("=== MiniGrid DRQN ===")
print(f"Env: {ENV_NAME}")
print(f"Obs: {'FullyObs' if FULLY_OBS else 'PartialObs (7x7)'}")
print()

# 環境セットアップ
env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
eval_env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
print(f"Observation shape: {env.obs_shape}, state_size: {env.state_size}")
print()

# DRQN Network作成
obs_dim = env.state_size  # 7*7*3 = 147 (部分観測) or 16*16*3 = 768 (完全観測)
action_size = 3  # left, right, forward
hidden_size = 64

qnet = DRQNNet(obs_dim, action_size, hidden_size)
target_qnet = DRQNNet(obs_dim, action_size, hidden_size)

# ダミー入力で重みを初期化（B, T, obs_dim）
dummy_input = Variable(np.zeros((1, 1, obs_dim), dtype=np.float32))
qnet(dummy_input)
target_qnet(dummy_input)

# LSTM状態をリセット
qnet.reset_state()
target_qnet.reset_state()

# DRQN エージェント
agent = DRQNAgent(
    qnet=qnet,
    target_qnet=target_qnet,
    action_size=action_size,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    lr=1e-3,
    batch_size=16,
    buffer_size=2000,  # エピソード数
    tau=0.005,
    warmup_episodes=30,
    seq_len=8,  # 学習用シーケンス長
    burn_in=4,  # LSTM安定化用
    grad_clip=1.0,
)

# %%
# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    agent=agent,
    num_episodes=500,
    eval_interval=50,
    eval_n=10,
    update_every=4,
    log_interval=50,
    stats_extractor=drqn_stats_extractor,
    eval_stats_extractor=minigrid_eval_stats_extractor,
)

# %%
result = trainer.train()

# %%
print()
print(f"Total episodes: {len(result.episode_rewards)}")
print(f"Final epsilon: {agent.epsilon:.4f}")
print(f"Episodes in buffer: {len(agent.buffer)}")

# %% [markdown]
# ## 学習曲線

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Episode Returns
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
axes[0].set_title("DRQN Episode Returns (Partial Obs)")
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
    render_env = SimpleActionWrapper(gym.make(ENV_NAME, render_mode="rgb_array"))
frames = []
action_names = ["left", "right", "forward"]
action_history = []


def process_obs_for_render(obs: dict) -> np.ndarray:
    """観測をフラット化"""
    image = obs["image"].flatten().astype(np.float32) / 10.0
    return image


# 評価時もLSTM状態をリセット
agent.reset_state()

obs, _ = render_env.reset()
obs_flat = process_obs_for_render(obs)

for step in range(100):
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
