# %% [markdown]
# # MiniGrid PPO
#
# PPO (Proximal Policy Optimization) を用いたMiniGrid環境の学習
#
# ## DQNとの違い
# - On-policy: 現在のポリシーで収集したデータのみ使用
# - Actor-Critic: π(a|s) と V(s) を同時に学習
# - クリッピング: 急激なポリシー更新を防止

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
from models.ppo import ActorCritic, PPOAgent, ppo_stats_extractor

# %% [markdown]
# ## 環境仕様
#
# | 項目 | 内容 |
# |------|------|
# | 環境 | MiniGrid-Empty-8x8-v0 |
# | 観測 | 完全観測: shape=(8, 8, 3), フラット化後192次元 |
# | 行動 | Discrete(3): left, right, forward |
# | 報酬 | ゴール到達時に 1 - 0.9 * (step_count / max_steps) |
#
# ## PPOのポイント
# - trajectory収集 → GAE計算 → ミニバッチ更新 → クリア
# - advantage = 「どれだけ良かったか」
# - clipping = 「保険」（急激な更新を防ぐ）


# %%
class SimpleActionWrapper(ActionWrapper):
    """Empty環境用にアクション空間を3に制限するラッパー"""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action


class MiniGridEnvWrapper:
    """MiniGrid用のラッパー"""

    def __init__(self, env_name: str = "MiniGrid-Empty-8x8-v0", fully_obs: bool = True):
        self.base_env = gym.make(env_name)
        if fully_obs:
            self.env = SimpleActionWrapper(FullyObsWrapper(self.base_env))
        else:
            self.env = SimpleActionWrapper(self.base_env)
        img_shape = self.env.observation_space["image"].shape
        assert img_shape is not None
        self.obs_shape = img_shape
        self.state_size = int(np.prod(img_shape))

    def _process_obs(self, obs: dict) -> np.ndarray:
        """imageをフラット化"""
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
# ## 実行：PPO（完全観測）

# %%
ENV_NAME = "MiniGrid-Empty-8x8-v0"
FULLY_OBS = True  # 完全観測モード

print("=== MiniGrid PPO ===")
print(f"Env: {ENV_NAME}")
print(f"Obs: {'FullyObs' if FULLY_OBS else 'PartialObs'}")
print()

# 環境セットアップ
env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
eval_env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
print(f"Observation shape: {env.obs_shape}, state_size: {env.state_size}")
print()

# ActorCritic ネットワーク作成
obs_dim = env.state_size
action_size = 3  # left, right, forward
hidden_size = 64

actor_critic = ActorCritic(obs_dim, action_size, hidden_size)

# ダミー入力で重みを初期化
dummy_input = Variable(np.zeros((1, obs_dim), dtype=np.float32))
actor_critic(dummy_input)

# PPO エージェント
agent = PPOAgent(
    actor_critic=actor_critic,
    action_size=action_size,
    obs_dim=obs_dim,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    lr=3e-4,
    n_epochs=4,
    batch_size=64,
    rollout_steps=512,  # 小さめに設定（MiniGridはエピソードが短い）
    max_grad_norm=0.5,
)

# %%
# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    agent=agent,
    eval_env=eval_env,
    num_episodes=500,
    eval_interval=50,
    eval_n=10,
    update_every=1,  # PPOは内部でrollout_stepsを管理
    log_interval=50,
    stats_extractor=ppo_stats_extractor,
    eval_stats_extractor=minigrid_eval_stats_extractor,
)

# %%
result = trainer.train()

# %%
print()
print(f"Total episodes: {len(result.episode_rewards)}")
print(f"Final policy loss: {agent.last_policy_loss:.4f}" if agent.last_policy_loss else "")
print(f"Final value loss: {agent.last_value_loss:.4f}" if agent.last_value_loss else "")
print(f"Final entropy: {agent.last_entropy:.4f}" if agent.last_entropy else "")

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
axes[0].set_title("PPO Episode Returns")
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


obs, _ = render_env.reset()
obs_flat = process_obs_for_render(obs)

for step in range(50):
    frame = render_env.render()
    frames.append(frame)

    action = agent.act(obs_flat, explore=False)
    action_history.append(action_names[action])
    obs, reward, terminated, truncated, _ = render_env.step(action)
    obs_flat = process_obs_for_render(obs)

    if terminated or truncated:
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
