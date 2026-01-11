# %% [markdown]
# # MiniGrid DQN
#
# DQNを用いたMiniGrid環境の学習
# FullyObsWrapper で完全観測モードを使用

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent import ReplayBuffer, BaseAgent
from dpl.agent_trainer import AgentTrainer, GymEnvWrapper, EvalResult

# %% [markdown]
# ## 環境仕様
#
# | 項目 | 内容 |
# |------|------|
# | 環境 | MiniGrid-Empty-5x5-v0 |
# | 観測 | shape=(5, 5, 3), 0-10のuint8 (FullyObsWrapper + ImgObsWrapper) |
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
    """MiniGrid用のCNN-based Q-Network

    画像観測 (B, H, W, C) を入力としてCNNで特徴抽出後、MLPヘッドでQ値を出力
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_size: int = 3,
        hidden_size: int = 128,
    ):
        super().__init__()
        H, W, C = obs_shape
        self.obs_shape = obs_shape
        self.action_size = action_size

        # CNN feature extractor
        self.cnn = L.Sequential(
            L.Conv2d(32, kernel_size=3, pad=1, in_channels=C),
            F.relu,
            L.Conv2d(64, kernel_size=3, pad=1),
            F.relu,
        )

        # Flatten後のサイズ: 64 * H * W
        flatten_size = 64 * H * W

        # MLP head
        self.head = L.Sequential(
            L.Linear(hidden_size, in_size=flatten_size),
            F.relu,
            L.Linear(action_size),
        )

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs

        # x: (B, H, W, C) → (B, C, H, W)
        x = F.transpose(x, 0, 3, 1, 2)

        # CNN feature extraction
        x = self.cnn(x)

        # Flatten: (B, C, H, W) → (B, C*H*W)
        batch_size = x.shape[0]
        x = F.reshape(x, (batch_size, -1))

        # MLP head
        return self.head(x)


# %%
class DQNAgent(BaseAgent):
    """Double DQN Agent for MiniGrid

    - Experience Replay
    - Target Network with soft update
    - Double DQN: action選択はqnet、Q値評価はtarget_qnetで行う
    - epsilon-greedy探索
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],  # (H, W, C)
        action_size: int = 3,  # left, right, forward for Empty environments
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 10000,
        tau: float = 0.005,
        hidden_size: int = 128,
        warmup_steps: int = 500,
    ):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.warmup_steps = warmup_steps

        # Q-Network (CNN-based)
        self.qnet = QNet(obs_shape, action_size, hidden_size)
        self.target_qnet = QNet(obs_shape, action_size, hidden_size)
        # ダミー入力で重みを初期化: (B, H, W, C)
        dummy_input = Variable(np.zeros((1, *obs_shape), dtype=np.float32))
        self.qnet(dummy_input)
        self.target_qnet(dummy_input)

        # Target networkを初期化
        self._hard_update_target()

        # Optimizer
        self.optimizer = Adam(lr=lr).setup(self.qnet)

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size)

        # 学習ステップカウンタ
        self.learn_step = 0

        # モニタリング用
        self.last_loss: float | None = None

    def _soft_update_target(self):
        """Target networkをsoft update: θ' ← τθ + (1-τ)θ'"""
        for main_param, target_param in zip(
            self.qnet.params(), self.target_qnet.params()
        ):
            if main_param.data is not None and target_param.data is not None:
                target_param.data = (
                    self.tau * main_param.data + (1 - self.tau) * target_param.data
                )

    def _hard_update_target(self):
        """Target networkをメインnetworkで完全に同期"""
        for main_param, target_param in zip(
            self.qnet.params(), self.target_qnet.params()
        ):
            if main_param.data is not None:
                target_param.data = main_param.data.copy()

    def _greedy_action(self, state: np.ndarray) -> int:
        """Q値が最大のアクションを選択"""
        # state: (H, W, C) → (1, H, W, C)
        state_v = Variable(state[np.newaxis, ...].astype(np.float32))
        q_values = self.qnet(state_v)
        return int(np.argmax(q_values.data_required))

    def get_action(self, state: np.ndarray) -> int:
        """epsilon-greedy戦略でアクションを選択"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return self._greedy_action(state)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """アクションを選択"""
        if explore:
            return self.get_action(state)
        return self._greedy_action(state)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        terminated=None,
    ):
        """経験をバッファに保存"""
        # (H, W, C) のまま保存（フラット化しない）
        self.buffer.push(state, action, reward, next_state, done)
        self.learn_step += 1

    def update(self) -> dict | None:
        """Replay Bufferからサンプリングして更新"""
        # Warmup期間中は更新しない
        if self.learn_step < self.warmup_steps:
            return None

        # バッファサイズが足りない場合は更新しない
        if len(self.buffer) < self.batch_size:
            return None

        # ミニバッチサンプリング
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_v = Variable(states)
        next_states_v = Variable(next_states)
        rewards_v = rewards.reshape(-1, 1)
        dones_v = dones.reshape(-1, 1)

        # Double DQN: action選択はqnet、Q値評価はtarget_qnet
        # 1. qnetで次状態のbest actionを選択
        next_q_values = self.qnet(next_states_v).data_required
        next_actions = np.argmax(next_q_values, axis=1)

        # 2. target_qnetでそのactionのQ値を評価
        next_q_target = self.target_qnet(next_states_v).data_required
        next_q = next_q_target[np.arange(self.batch_size), next_actions].reshape(-1, 1)

        # TD target: r + γ * Q_target(s', argmax_a Q(s', a))
        td_target = rewards_v + self.gamma * next_q * (1 - dones_v)

        # 現在のQ値
        current_q = self.qnet(states_v)
        # 選択したアクションのQ値のみ取り出す
        actions_idx = actions.astype(np.int64)
        current_q_selected = Variable(
            current_q.data_required[np.arange(self.batch_size), actions_idx].reshape(
                -1, 1
            )
        )

        # MSE Loss
        td_target_v = Variable(td_target.astype(np.float32))
        loss = ((current_q_selected - td_target_v) ** 2).sum() / self.batch_size

        # 勾配計算と更新
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        # Target networkのsoft update
        self._soft_update_target()

        self.last_loss = float(loss.data_required)
        return {"loss": self.last_loss}

    def decay_epsilon(self):
        """エピソード終了時にepsilonを減衰"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# %% [markdown]
# ## MiniGrid用ラッパー


# %%
class MiniGridEnvWrapper:
    """MiniGrid用のラッパー

    FullyObsWrapper + ImgObsWrapper を適用
    観測は (H, W, C) 形式のまま保持（CNNに渡すため）
    """

    def __init__(self, env_name: str = "MiniGrid-Empty-5x5-v0"):
        self.base_env = gym.make(env_name)
        self.env = ImgObsWrapper(FullyObsWrapper(self.base_env))
        shape = self.env.observation_space.shape
        assert shape is not None and len(shape) == 3
        self.obs_shape: tuple[int, int, int] = (
            shape[0],
            shape[1],
            shape[2],
        )  # (H, W, C)

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        return obs.astype(np.float32) / 10.0  # 正規化、shape=(H, W, C)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs.astype(np.float32) / 10.0,  # 正規化、shape=(H, W, C)
            float(reward),
            terminated,
            truncated,
            info,
        )


# %% [markdown]
# ## 統計抽出関数


# %%
def minigrid_stats_extractor(agent) -> str | None:
    """MiniGrid Agent用の統計抽出関数"""
    epsilon = getattr(agent, "epsilon", None)
    loss = getattr(agent, "last_loss", None)

    parts = []
    if epsilon is not None:
        parts.append(f"ε={epsilon:.3f}")
    if loss is not None:
        parts.append(f"L={loss:.4f}")

    if not parts:
        return None
    return ", " + ", ".join(parts)


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
print("=== MiniGrid-Empty-5x5-v0 DQN ===")
print()

# 環境セットアップ
env = MiniGridEnvWrapper("MiniGrid-Empty-5x5-v0")
print(f"Observation shape: {env.obs_shape}")
print()

# DQN エージェント (CNN-based)
agent = DQNAgent(
    obs_shape=env.obs_shape,  # (H, W, C)
    action_size=3,  # left, right, forward（Empty環境用）
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    lr=1e-3,
    batch_size=64,
    buffer_size=10000,
    tau=0.005,
    hidden_size=128,
    warmup_steps=500,
)

# %%
# AgentTrainerで学習
trainer = AgentTrainer(
    env=env,
    agent=agent,
    num_episodes=300,
    eval_interval=30,
    eval_n=10,
    update_every=1,
    log_interval=30,
    stats_extractor=minigrid_stats_extractor,
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
render_env = ImgObsWrapper(
    FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array"))
)
frames = []
action_names = ["left", "right", "forward"]
action_history = []

obs, _ = render_env.reset()
obs_normalized = obs.astype(np.float32) / 10.0  # (H, W, C)

for step in range(50):
    # rgb_array でレンダリング
    frame = render_env.render()
    frames.append(frame)

    action = agent.act(obs_normalized, explore=False)
    action_history.append(action_names[action])
    obs, reward, terminated, truncated, _ = render_env.step(action)
    obs_normalized = obs.astype(np.float32) / 10.0  # (H, W, C)

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
