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

from dpl.agent_trainer import AgentTrainer, GymEnvWrapper, EvalResult
from models.sac import SACAgent, sac_stats_extractor

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


# %% [markdown]
# ## 統計抽出関数


# %%
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
    stats_extractor=sac_stats_extractor,
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
