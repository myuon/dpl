# %% [markdown]
# # Pendulum-v1 強化学習
#
# REINFORCE（モンテカルロ方策勾配）による連続行動空間の学習
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
# ## REINFORCE with Baseline（Advantage Actor-Critic）
#
# Baseline（価値関数 V(s)）付きの方策勾配法。
#
# - Actor（方策）: π(a|s) = N(μ(s), σ(s))
# - Critic（価値関数）: V(s) → 状態の期待リターンを推定
# - Advantage: A_t = G_t - V(s_t)
# - Actor損失: -E[log π(a|s) * A_t]
# - Critic損失: MSE(V(s), G_t)
#
# 役割分担:
# - Actor: "どの行動が期待より良かったか" を advantage で学ぶ
# - Critic: "その状態の期待値" を回帰で学ぶ


# %%
from dpl.layers import Layer, Sequential


class GaussianPolicy(Layer):
    """ガウス方策ネットワーク

    状態 s → (μ, σ) を出力
    Layerを継承して params()/cleargrads() を利用
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        action_scale: float = 2.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共有層
        self.shared = Sequential(
            L.Linear(hidden_dim, in_size=state_dim),
            F.relu,
            L.Linear(hidden_dim),
            F.relu,
        )

        # μ と log_σ を別々に出力
        self.mean_head = L.Linear(action_dim)
        self.log_std_head = L.Linear(action_dim)

    def predict(self, state: Variable) -> tuple[Variable, Variable]:
        """状態から平均と標準偏差を出力"""
        h = self.shared.apply(state)

        mean = self.mean_head.apply(h)
        log_std = self.log_std_head.apply(h)

        # log_std をクランプ
        log_std_data = np.clip(
            log_std.data_required, self.log_std_min, self.log_std_max
        )
        log_std = Variable(log_std_data)
        std = F.exp(log_std)

        return mean, std


# %%
class ValueNetwork(Layer):
    """価値関数ネットワーク（Critic）

    状態 s → V(s) を出力
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = Sequential(
            L.Linear(hidden_dim, in_size=state_dim),
            F.relu,
            L.Linear(hidden_dim),
            F.relu,
            L.Linear(1),  # スカラー出力
        )

    def predict(self, state: Variable) -> Variable:
        """状態から価値を出力"""
        return self.net.apply(state)


# %%
class ReinforceAgent(BaseAgent):
    """REINFORCE エージェント

    モンテカルロ方策勾配。エピソード終了後にまとめて更新。
    dpl（自作NNライブラリ）を使用。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 64,
        action_scale: float = 2.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.gamma = gamma
        self.action_scale = action_scale

        # Actor（方策ネットワーク）
        self.policy = GaussianPolicy(
            state_dim, action_dim, hidden_dim, action_scale
        )
        self.actor_optimizer = Adam(lr=lr).setup(self.policy)

        # Critic（価値関数ネットワーク）
        self.critic = ValueNetwork(state_dim, hidden_dim)
        self.critic_optimizer = Adam(lr=lr).setup(self.critic)

        # エピソード内のトラジェクトリ
        self.episode_states: list[np.ndarray] = []
        self.episode_actions: list[np.ndarray] = []
        self.episode_rewards: list[float] = []

        # ダミーバッファ（インターフェース互換用）
        self.buffer = ReplayBuffer(capacity=1, continuous_action=True)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, explore=True)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_v = Variable(state.reshape(1, -1).astype(np.float32))
        mean, std = self.policy.predict(state_v)

        if explore:
            # サンプリング
            eps = np.random.randn(*mean.shape).astype(np.float32)
            raw_action = mean.data_required + std.data_required * eps
        else:
            raw_action = mean.data_required

        # tanh スケーリング
        action = self.action_scale * np.tanh(raw_action)
        return action.flatten().astype(np.float32)

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
        # state と action を保存（log_prob は後で計算）
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def update(self) -> dict | None:
        """エピソード終了時に呼ばれる想定だが、
        AgentTrainer はステップごとに呼ぶので、
        end_episode で実際の更新を行う"""
        return None

    def start_episode(self):
        """エピソード開始時にトラジェクトリをクリア"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def _compute_log_prob(
        self, states: Variable, actions: np.ndarray
    ) -> Variable:
        """行動に対する log_prob を計算（tanh 補正込み）"""
        mean, std = self.policy.predict(states)

        # action から raw_action を逆算
        # action = action_scale * tanh(raw) → raw = atanh(action / action_scale)
        # 数値安定性のため clamp
        action_normalized = np.clip(actions / self.action_scale, -0.999, 0.999)
        raw_action = np.arctanh(action_normalized)
        raw_action_v = Variable(raw_action.astype(np.float32))

        # log_prob 計算: log N(raw | mean, std)
        # = -0.5 * log(2π) - log(std) - 0.5 * ((raw - mean) / std)^2
        diff = raw_action_v - mean
        log_prob = -0.5 * np.log(2 * np.pi) - F.log(std) - 0.5 * (diff / std) ** 2

        # tanh の jacobian 補正: -log(action_scale * (1 - tanh(raw)^2))
        tanh_raw = F.tanh(raw_action_v)
        jacobian_correction = F.log(
            Variable(np.full_like(raw_action, self.action_scale, dtype=np.float32))
            * (Variable(np.ones_like(raw_action, dtype=np.float32)) - tanh_raw * tanh_raw)
            + Variable(np.full_like(raw_action, 1e-6, dtype=np.float32))
        )
        log_prob = log_prob - jacobian_correction

        # 行動次元で sum
        log_prob = log_prob.sum(axis=1)

        return log_prob

    def end_episode(self):
        """エピソード終了時にActor/Criticを更新"""
        if len(self.episode_rewards) == 0:
            return

        # モンテカルロリターンを計算
        returns = []
        G = 0.0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns, dtype=np.float32)
        returns_v = Variable(returns.reshape(-1, 1))

        # states, actions をテンソルに変換
        states = Variable(np.array(self.episode_states, dtype=np.float32))
        actions = np.array(self.episode_actions, dtype=np.float32)

        # Critic: V(s) を計算
        values = self.critic.predict(states)  # shape: (T, 1)

        # Advantage: A_t = G_t - V(s_t)
        # .data_required で計算グラフから切り離す（Actor更新時にCriticの勾配が流れないように）
        advantages = returns_v.data_required - values.data_required
        advantages = advantages.reshape(-1)

        # Advantageの正規化（分散削減）
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_v = Variable(advantages.astype(np.float32))

        # --- Actor更新 ---
        log_probs = self._compute_log_prob(states, actions)
        actor_loss = -(log_probs * advantages_v).sum() / len(returns)

        self.policy.cleargrads()
        actor_loss.backward()
        self.actor_optimizer.update()

        # --- Critic更新 ---
        # 損失: MSE(V(s), G_t)
        # 新しくforwardしてグラフを作り直す
        values = self.critic.predict(states)
        critic_loss = ((values - returns_v) ** 2).sum() / len(returns)

        self.critic.cleargrads()
        critic_loss.backward()
        self.critic_optimizer.update()

        # トラジェクトリをクリア
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []


# %% [markdown]
# ## 実行：REINFORCE with Baseline

# %%
print("=== Pendulum-v1 REINFORCE with Baseline (dpl) ===")
print()

# REINFORCE with Baseline エージェント
agent = ReinforceAgent(
    state_dim=3,
    action_dim=1,
    hidden_dim=64,
    action_scale=2.0,
    lr=3e-4,
    gamma=0.99,
)

# %%
# 環境をラップ
env = GymEnvWrapper(gym.make("Pendulum-v1"))
eval_env = GymEnvWrapper(gym.make("Pendulum-v1"))

# AgentTrainerで学習
# REINFORCEはエピソード単位で更新するので update_every は大きくてOK
trainer = AgentTrainer(
    env=env,
    eval_env=eval_env,
    agent=agent,
    num_episodes=500,
    eval_interval=50,
    eval_n=10,
    update_every=9999,  # end_episode で更新するので使わない
    log_interval=50,
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
# 移動平均
window = 20
if len(result.episode_rewards) >= window:
    moving_avg = np.convolve(
        result.episode_rewards, np.ones(window) / window, mode="valid"
    )
    axes[0].plot(
        range(window - 1, len(result.episode_rewards)), moving_avg, label=f"MA({window})"
    )
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Return")
axes[0].set_title("Episode Returns")
axes[0].legend()
axes[0].grid(True)

# Eval Returns
if result.eval_returns:
    episodes, means = zip(*result.eval_returns)
    axes[1].plot(episodes, means, marker="o")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Eval Return")
    axes[1].set_title("Evaluation Returns (explore=False)")
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

    # explore=False で決定的行動を確認
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
