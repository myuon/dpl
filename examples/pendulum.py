# %% [markdown]
# # Pendulum-v1 強化学習
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
# | 観測 | shape=(3,), [cos(θ), sin(θ), angular_velocity] |
# | 観測範囲 | [-1, -1, -8] ~ [1, 1, 8] |
# | 行動 | shape=(1,), 連続値トルク |
# | 行動範囲 | [-2, 2] |
# | 報酬 | 常に負（-16.27 ~ 0）、振り子が上向きで0に近づく |
# | terminated | 常にFalse |
# | truncated | 200ステップでTrue |

# %% [markdown]
# ## SAC（Soft Actor-Critic）
#
# 最大エントロピー強化学習。探索を"ノイズで足す"から"目的関数で制御する"へ。
#
# ## SACの特徴（TD3からの改良点）
# 1. **確率方策（Stochastic Policy）**: π(a|s) = tanh(μ(s) + σ(s) * ε)
#    → 探索がポリシー自体に組み込まれる
# 2. **エントロピー正則化**: 目的 = E[Q(s,a) - α * log π(a|s)]
#    → "高得点を取りつつ、不確実性も残す"
# 3. **Reparameterization Trick**: a = tanh(μ + σ * ε)で勾配を流す
# 4. **自動温度調整**: αを学習して目標エントロピーを維持
#
# ## 更新式
# - Target: y = r + γ * (min(Q1'(s',a'), Q2'(s',a')) - α * log π(a'|s'))
# - Critic: L = E[(Q(s,a) - y)²]
# - Actor: L = E[α * log π(a|s) - Q(s, a)]  （エントロピー付き方策勾配）
# - α: L = -α * E[log π(a|s) + target_entropy]


# %% [markdown]
# ## 統計抽出関数


# %%
def pendulum_eval_stats_extractor(result: EvalResult) -> str:
    """Pendulum評価結果の表示"""
    return f"Return={result.avg_return:.2f}, Mean|Action|={result.mean_abs_action:.3f}"


# %% [markdown]
# ## 実行：SAC

# %%
print("=== Pendulum-v1 SAC (dpl) ===")
print()

# SAC エージェント
agent = SACAgent(
    state_dim=3,
    action_dim=1,
    hidden_dim=256,
    action_scale=2.0,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    buffer_size=100000,
    batch_size=256,
    warmup_steps=1000,
    # SAC固有
    init_alpha=0.2,
    auto_alpha=True,
    target_entropy=-1.0,  # -action_dim
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
    stats_extractor=sac_stats_extractor,
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
axes[0].set_title("SAC Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("SAC Evaluation Returns (explore=False)")
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
