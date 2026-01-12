# %% [markdown]
# # MiniGrid PPO + RNN
#
# PPO + LSTM を用いた部分観測MiniGrid環境の学習
#
# ## PPO vs PPO+RNN の違い
# | 項目 | PPO | PPO+RNN |
# |------|-----|---------|
# | バッファ | RolloutBuffer (flat) | EpisodeRolloutBuffer (episode単位) |
# | ネットワーク | MLP | MLP → LSTM |
# | ミニバッチ | ランダムシャッフル | シーケンス単位 |
# | 状態管理 | なし | reset_state, start/end_episode |
#
# ## PPO+RNN 特有の罠と対策
# | 罠 | 対策 |
# |----|------|
# | Hidden state リーク | Episode boundary で reset |
# | Minibatch 分割 | Sequence を切らない |
# | Entropy 崩壊 | entropy_coef=0.02（高め） |

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
from dpl.agent_trainer import EvalResult
from models.ppo_rnn import RecurrentActorCritic, PPORNNAgent, ppo_rnn_stats_extractor

# %% [markdown]
# ## 環境仕様
#
# | 項目 | 内容 |
# |------|------|
# | 環境 | MiniGrid-Empty-8x8-v0 |
# | 観測 | **部分観測**: shape=(7, 7, 3), エージェント視点の7x7グリッド |
# | 行動 | Discrete(3): left, right, forward |
# | 報酬 | ゴール到達時に 1 - 0.9 * (step_count / max_steps) |
#
# ## PPO+RNN のポイント
# - obs_t → MLP → feature_t → LSTM (with h_{t-1}) → h_t → (π, V)
# - エピソード単位でバッファに保存
# - シーケンス単位でミニバッチ更新（burn-in付き）


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

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
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


class VectorEnvWrapper:
    """複数環境を並列実行するラッパー

    MiniGridEnvWrapperを複数インスタンス化して並列stepを実行
    """

    def __init__(self, env_fn, n_envs: int = 8):
        self.n_envs = n_envs
        self.envs = [env_fn() for _ in range(n_envs)]
        self.obs_shape = self.envs[0].obs_shape
        self.state_size = self.envs[0].state_size

    def reset(self) -> np.ndarray:
        """全環境リセット → (n_envs, obs_dim)"""
        return np.stack([env.reset() for env in self.envs])

    def reset_single(self, env_id: int) -> np.ndarray:
        """特定環境のみリセット → (obs_dim,)"""
        return self.envs[env_id].reset()

    def step(self, actions: np.ndarray):
        """全環境ステップ

        Args:
            actions: (n_envs,)

        Returns:
            obs: (n_envs, obs_dim)
            rewards: (n_envs,)
            terminateds: (n_envs,)
            truncateds: (n_envs,)
        """
        obs_list, rewards, terminateds, truncateds = [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = env.step(int(a))
            obs_list.append(o)
            rewards.append(r)
            terminateds.append(term)
            truncateds.append(trunc)
        return (
            np.stack(obs_list),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
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
# ## 実行：PPO+RNN（部分観測, n_envs=8）

# %%
ENV_NAME = "MiniGrid-Empty-8x8-v0"
FULLY_OBS = False  # 部分観測モード（RNNの効果を確認）
N_ENVS = 8  # 並列環境数
ROLLOUT_LEN = 256  # 各環境でのrollout長（8 × 256 = 2048 steps/update）

print("=== MiniGrid PPO+RNN (Parallel Envs) ===")
print(f"Env: {ENV_NAME}")
print(f"Obs: {'FullyObs' if FULLY_OBS else 'PartialObs (7x7)'}")
print(f"n_envs: {N_ENVS}, rollout_len: {ROLLOUT_LEN}")
print()

# 並列環境セットアップ
env = VectorEnvWrapper(
    env_fn=lambda: MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS),
    n_envs=N_ENVS,
)
# 評価用は単一環境
eval_env = MiniGridEnvWrapper(ENV_NAME, fully_obs=FULLY_OBS)
print(f"Observation shape: {env.obs_shape}, state_size: {env.state_size}")
print()

# RecurrentActorCritic作成（n_envs対応）
obs_dim = env.state_size  # 7*7*3 = 147 (部分観測)
action_size = 3  # left, right, forward
hidden_size = 64

actor_critic = RecurrentActorCritic(obs_dim, action_size, hidden_size, n_envs=N_ENVS)

# ダミー入力で重みを初期化（B, T, obs_dim）
dummy_input = Variable(np.zeros((1, 1, obs_dim), dtype=np.float32))
actor_critic(dummy_input)

# LSTM状態をリセット
actor_critic.reset_state()

# PPO+RNN エージェント（n_envs対応）
agent = PPORNNAgent(
    actor_critic=actor_critic,
    action_size=action_size,
    obs_dim=obs_dim,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.005,  # 安定性のため調整
    value_coef=0.5,
    lr=3e-4,
    n_epochs=3,
    seq_len=16,
    burn_in=4,
    batch_size=32,
    rollout_steps=N_ENVS * ROLLOUT_LEN,  # 2048 steps/update
    max_episodes=200,
    max_grad_norm=0.5,
    target_kl=0.05,
    n_envs=N_ENVS,
    rollout_len=ROLLOUT_LEN,
)

# %% [markdown]
# ## 学習ループ（並列環境 + collect_rollout）

# %%
import time

NUM_UPDATES = 200  # 更新回数
EVAL_INTERVAL = 20  # 評価間隔（更新回数）
EVAL_N = 100  # 評価エピソード数
LOG_INTERVAL = 10  # ログ間隔

# 結果記録用
all_episode_rewards: list[float] = []
eval_returns: list[tuple[int, float]] = []
eval_success_rates: list[tuple[int, float]] = []

start_time = time.time()

for update in range(NUM_UPDATES):
    # rollout収集 & 更新
    steps, episode_rewards = agent.collect_rollout(env)
    all_episode_rewards.extend(episode_rewards)

    loss_dict = agent.update()

    # ログ出力
    if (update + 1) % LOG_INTERVAL == 0:
        elapsed = time.time() - start_time
        remaining = elapsed / (update + 1) * (NUM_UPDATES - update - 1)

        avg_return = np.mean(episode_rewards) if episode_rewards else 0.0
        stats = ppo_rnn_stats_extractor(agent) or ""
        print(
            f"Update {update + 1}/{NUM_UPDATES}: "
            f"Episodes={len(episode_rewards)}, AvgReturn={avg_return:.2f}{stats}, "
            f"ETA={remaining:.0f}s"
        )

    # 評価
    if (update + 1) % EVAL_INTERVAL == 0:
        # 評価用に単一環境モードのactor_criticを作成
        eval_actor_critic = RecurrentActorCritic(obs_dim, action_size, hidden_size, n_envs=1)
        # 重みをコピー
        for src, dst in zip(actor_critic.params(), eval_actor_critic.params()):
            if src.data is not None:
                dst.data = src.data.copy()

        returns = []
        successes = []
        for ep in range(EVAL_N):
            eval_actor_critic.reset_state()
            obs = eval_env.reset(seed=42 + ep)
            done = False
            total_return = 0.0
            while not done:
                logits, _ = eval_actor_critic.step(obs)
                logits_max = np.max(logits)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / np.sum(exp_logits)
                action = np.random.choice(action_size, p=probs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_return += reward
            returns.append(total_return)
            successes.append(terminated)

        avg_return = float(np.mean(returns))
        success_rate = float(np.mean(successes))
        eval_returns.append((update + 1, avg_return))
        eval_success_rates.append((update + 1, success_rate))
        print(
            f"  → Eval: Return={avg_return:.3f}, "
            f"Success={success_rate*100:.0f}%"
        )

print(f"\nTraining completed in {time.time() - start_time:.1f}s")

# %%
print()
print(f"Total episodes: {len(all_episode_rewards)}")
print(
    f"Final policy loss: {agent.last_policy_loss:.4f}" if agent.last_policy_loss else ""
)
print(f"Final value loss: {agent.last_value_loss:.4f}" if agent.last_value_loss else "")
print(f"Final entropy: {agent.last_entropy:.4f}" if agent.last_entropy else "")

# %% [markdown]
# ## 学習曲線

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Episode Returns
axes[0].plot(all_episode_rewards, alpha=0.3, label="Episode")
window = 50
if len(all_episode_rewards) >= window:
    moving_avg = np.convolve(
        all_episode_rewards, np.ones(window) / window, mode="valid"
    )
    axes[0].plot(
        range(window - 1, len(all_episode_rewards)),
        moving_avg,
        label=f"MA({window})",
    )
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Return")
axes[0].set_title("PPO+RNN Episode Returns (Partial Obs, n_envs=8)")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if eval_returns:
    updates, means = zip(*eval_returns)
    axes[1].plot(updates, means, marker="o")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("Evaluation Returns")
    axes[1].grid(True)

# Success Rate
if eval_success_rates:
    updates, rates = zip(*eval_success_rates)
    axes[2].plot(updates, [r * 100 for r in rates], marker="o", color="green")
    axes[2].set_xlabel("Update")
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


# 評価用にn_envs=1のactor_criticを作成
render_actor_critic = RecurrentActorCritic(obs_dim, action_size, hidden_size, n_envs=1)
for src, dst in zip(actor_critic.params(), render_actor_critic.params()):
    if src.data is not None:
        dst.data = src.data.copy()
render_actor_critic.reset_state()

obs, _ = render_env.reset()
obs_flat = process_obs_for_render(obs)

for step in range(100):
    # rgb_array でレンダリング
    frame = render_env.render()
    frames.append(frame)

    # render_actor_criticを使って行動選択
    logits, _ = render_actor_critic.step(obs_flat)
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    action = np.random.choice(action_size, p=probs)
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
