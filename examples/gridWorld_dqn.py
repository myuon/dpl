# %% [markdown]
# # GridWorld DQN
# DQNを用いたGridWorldの学習

# %% Imports
import numpy as np

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent_trainer import AgentTrainer
from dpl.agent import ReplayBuffer, BaseAgent


# %% GridWorld Environment
class GridWorld:
    """シンプルなGridWorld環境

    5x5のグリッドで、エージェントがスタート地点(0,0)からゴール地点(4,4)まで移動する。
    障害物も配置可能。

    行動:
        0: 上 (y-1)
        1: 下 (y+1)
        2: 左 (x-1)
        3: 右 (x+1)

    報酬:
        ゴール到達: +1.0
        壁にぶつかる: -0.1
        各ステップ: -0.01 (効率的な経路を学習させるため)
    """

    def __init__(self, size: int = 5, max_steps: int = 100):
        self.size = size
        self.max_steps = max_steps
        self.action_space_n = 4  # 上下左右

        # 障害物の位置（固定）
        self.obstacles = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)}

        # ゴール位置
        self.goal = (size - 1, size - 1)

        # スタート位置
        self.start = (0, 0)

        self.reset()

    def reset(self) -> np.ndarray:
        """環境をリセット"""
        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """状態を返す（正規化された座標）"""
        # 座標を0-1に正規化
        return np.array(
            [self.agent_pos[0] / (self.size - 1), self.agent_pos[1] / (self.size - 1)],
            dtype=np.float32,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """1ステップ実行

        Args:
            action: 行動（0:上, 1:下, 2:左, 3:右）

        Returns:
            observation: 新しい状態
            reward: 報酬
            terminated: ゴール到達でTrue
            truncated: 最大ステップ数超過でTrue
            info: 追加情報
        """
        self.steps += 1

        # 移動先を計算
        new_pos = self.agent_pos.copy()
        if action == 0:  # 上
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # 下
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 2:  # 左
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # 右
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)

        # 壁チェック（移動しなかった場合）
        hit_wall = new_pos == self.agent_pos

        # 障害物チェック
        if tuple(new_pos) in self.obstacles:
            # 障害物には移動できない
            hit_wall = True
        else:
            self.agent_pos = new_pos

        # 報酬計算
        reward = -0.01  # ステップペナルティ
        if hit_wall:
            reward -= 0.1  # 壁ペナルティ

        # ゴール判定
        terminated = tuple(self.agent_pos) == self.goal
        if terminated:
            reward = 1.0

        # 最大ステップ超過
        truncated = self.steps >= self.max_steps

        return self._get_state(), reward, terminated, truncated, {}

    def render(self):
        """グリッドを表示"""
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        # 障害物
        for ox, oy in self.obstacles:
            grid[oy][ox] = "#"

        # ゴール
        gx, gy = self.goal
        grid[gy][gx] = "G"

        # エージェント
        ax, ay = self.agent_pos
        grid[ay][ax] = "A"

        print("-" * (self.size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.size * 2 + 1))


# %% Q-Network
class QNet(L.Sequential):
    """GridWorld用のQ-Network

    入力: 状態(2次元: x座標、y座標、正規化済み)
    出力: 各アクションのQ値(4次元: 上、下、左、右)
    """

    def __init__(
        self, state_size: int = 2, action_size: int = 4, hidden_size: int = 64
    ):
        super().__init__(
            L.Linear(hidden_size),
            F.relu,
            L.Linear(hidden_size),
            F.relu,
            L.Linear(action_size),
        )
        self.state_size = state_size
        self.action_size = action_size


# %% DQN Agent
class DQNAgent(BaseAgent):
    """Double DQN Agent for GridWorld

    - Experience Replay: 過去の経験をバッファに保存し、ミニバッチで学習
    - Target Network: 安定した学習のために定期的にコピー
    - Double DQN: action選択はqnet、Q値評価はtarget_qnetで行う
    - MSE loss
    - Gradient clipping: 勾配爆発を防ぐ
    - epsilon-greedy探索
    """

    def __init__(
        self,
        state_size: int = 2,
        action_size: int = 4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,  # per episode
        lr: float = 1e-3,
        batch_size: int = 32,
        buffer_size: int = 10000,
        tau: float = 0.005,  # soft update coefficient
        target_update_freq: int = 100,
        hidden_size: int = 64,
        grad_clip: float = 1.0,
        warmup_steps: int = 500,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps

        # Q-Network
        self.qnet = QNet(state_size, action_size, hidden_size)
        self.target_qnet = QNet(state_size, action_size, hidden_size)

        # ダミー入力で重みを初期化
        dummy_input = Variable(np.zeros((1, state_size), dtype=np.float32))
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

    def _soft_update_target(self):
        """Target networkをsoft updateで更新: θ' ← τθ + (1-τ)θ'"""
        for main_layer, target_layer in zip(self.qnet.layers, self.target_qnet.layers):
            if isinstance(main_layer, L.Linear) and isinstance(target_layer, L.Linear):
                target_layer.W.data = (
                    self.tau * main_layer.W.data + (1 - self.tau) * target_layer.W.data
                )
                if main_layer.b is not None:
                    target_layer.b.data = (
                        self.tau * main_layer.b.data
                        + (1 - self.tau) * target_layer.b.data
                    )

    def _hard_update_target(self):
        """Target networkをメインnetworkで完全に同期"""
        for main_layer, target_layer in zip(self.qnet.layers, self.target_qnet.layers):
            if isinstance(main_layer, L.Linear):
                target_layer.W.data = main_layer.W.data.copy()
                if main_layer.b is not None:
                    target_layer.b.data = main_layer.b.data.copy()

    def _clip_grads(self):
        """Gradient clipping (global norm)"""
        grads = []
        for param in self.qnet.params():
            if param.grad is not None:
                grads.append(param.grad.data_required.flatten())

        if not grads:
            return

        all_grads = np.concatenate(grads)
        global_norm = np.sqrt(np.sum(all_grads**2))

        if global_norm > self.grad_clip:
            scale = self.grad_clip / global_norm
            for param in self.qnet.params():
                if param.grad is not None:
                    param.grad.data = param.grad.data_required * scale

    def get_action(self, state: np.ndarray) -> int:
        """epsilon-greedy戦略でアクションを選択"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return self._greedy_action(state)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """指定したepsilonでアクションを選択"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray) -> int:
        """Q値が最大のアクションを選択"""
        state_var = Variable(state.reshape(1, -1).astype(np.float32))
        q_values = self.qnet(state_var)
        return int(np.argmax(q_values.data_required[0]))

    def store(self, state, action, reward, next_state, done):
        """経験をバッファに保存"""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """ミニバッチでQ-Networkを更新

        Returns:
            loss値（バッファが足りない場合やwarmup中はNone）
        """
        if len(self.buffer) < self.warmup_steps:
            return None
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Double DQN
        next_states_var = Variable(next_states)

        next_q_main = self.qnet(next_states_var).data_required
        best_actions = np.argmax(next_q_main, axis=1)

        next_q_target = self.target_qnet(next_states_var).data_required
        max_next_q = next_q_target[np.arange(len(best_actions)), best_actions]

        targets = rewards + self.gamma * max_next_q * (1 - dones)

        states_var = Variable(states)
        q_values = self.qnet(states_var)

        action_masks = np.eye(self.action_size)[actions]
        current_q = F.sum(q_values * Variable(action_masks.astype(np.float32)), axis=1)

        targets_var = Variable(targets.astype(np.float32))
        loss = F.mean_squared_error(current_q, targets_var)

        self.qnet.cleargrads()
        loss.backward()

        self._clip_grads()

        self.optimizer.update()

        self.learn_step += 1
        if self.tau > 0:
            self._soft_update_target()
        else:
            if self.learn_step % self.target_update_freq == 0:
                self._hard_update_target()

        return float(loss.data_required)

    def decay_epsilon(self):
        """エピソード終了時にepsilonを減衰"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# %% Evaluation Function
def evaluate_agent(agent: DQNAgent, num_episodes: int = 3):
    """学習済みエージェントをrenderして評価"""
    env = GridWorld()

    for episode in range(num_episodes):
        observation = env.reset()
        total_reward = 0
        steps = 0

        print(f"\n=== Evaluation Episode {episode + 1} ===")
        env.render()

        while True:
            state_var = Variable(observation.reshape(1, -1).astype(np.float32))
            q_values = agent.qnet(state_var)
            action = int(np.argmax(q_values.data_required[0]))

            action_names = ["上", "下", "左", "右"]
            print(f"Step {steps + 1}: Action = {action_names[action]}")

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if terminated:
                print(f"ゴール到達! Total Reward = {total_reward:.2f}, Steps = {steps}")
                break
            if truncated:
                print(f"時間切れ... Total Reward = {total_reward:.2f}, Steps = {steps}")
                break


# %% Training
print("Training DQN Agent on GridWorld...")

env = GridWorld()
eval_env = GridWorld()
agent = DQNAgent()

trainer = AgentTrainer(
    env=env,
    eval_env=eval_env,
    agent=agent,
    num_episodes=500,
    eval_interval=50,
)

result = trainer.train()

# %% Visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(result.episode_rewards)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("DQN: Episode Rewards (GridWorld)")
ax1.grid(True)

ax2.plot(result.episode_losses)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Average Loss")
ax2.set_title("DQN: Average Loss per Episode")
ax2.grid(True)

if result.eval_returns:
    episodes, returns = zip(*result.eval_returns)
    ax3.plot(episodes, returns, marker="o", markersize=8, linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Eval Return")
    ax3.set_title("DQN: Eval Return (ε=0, n=20)")
    ax3.grid(True)

plt.tight_layout()
plt.show()


# %% Q-Value Heatmap
def plot_q_heatmap(agent: DQNAgent, grid_size: int = 5):
    """各セルのQ値を4方向の三角形で可視化

    各セルを対角線で4つの三角形に分割:
    - 上: 上方向への移動のQ値
    - 下: 下方向への移動のQ値
    - 左: 左方向への移動のQ値
    - 右: 右方向への移動のQ値
    """
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors

    # 各セルのQ値を計算
    q_values = np.zeros((grid_size, grid_size, 4))

    for y in range(grid_size):
        for x in range(grid_size):
            state = np.array(
                [x / (grid_size - 1), y / (grid_size - 1)], dtype=np.float32
            )
            state_var = Variable(state.reshape(1, -1))
            q = agent.qnet(state_var).data_required[0]
            q_values[y, x] = q

    # 障害物とゴール
    obstacles = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)}
    goal = (grid_size - 1, grid_size - 1)

    # Q値の範囲を取得（カラーマップ用）
    q_min = np.min(q_values)
    q_max = np.max(q_values)
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.get_cmap("RdYlGn")

    _, ax = plt.subplots(figsize=(10, 10))

    # 各セルに4つの三角形を描画
    for y in range(grid_size):
        for x in range(grid_size):
            # セルの中心
            cx, cy = x, y

            # 4つの三角形の頂点（セルの角と中心）
            # 上三角形 (action=0: 上)
            top_tri = [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5), (cx, cy)]
            # 下三角形 (action=1: 下)
            bottom_tri = [(cx - 0.5, cy + 0.5), (cx + 0.5, cy + 0.5), (cx, cy)]
            # 左三角形 (action=2: 左)
            left_tri = [(cx - 0.5, cy - 0.5), (cx - 0.5, cy + 0.5), (cx, cy)]
            # 右三角形 (action=3: 右)
            right_tri = [(cx + 0.5, cy - 0.5), (cx + 0.5, cy + 0.5), (cx, cy)]

            triangles = [top_tri, bottom_tri, left_tri, right_tri]
            action_labels = ["↑", "↓", "←", "→"]
            # ラベルの位置オフセット
            label_offsets = [(0, -0.25), (0, 0.25), (-0.25, 0), (0.25, 0)]

            if (x, y) in obstacles:
                # 障害物はグレーで塗りつぶし
                for tri in triangles:
                    poly = Polygon(
                        tri, facecolor="gray", edgecolor="black", linewidth=0.5
                    )
                    ax.add_patch(poly)
                ax.text(
                    cx,
                    cy,
                    "#",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                # 各方向のQ値で色付け
                for i, (tri, _, offset) in enumerate(
                    zip(triangles, action_labels, label_offsets)
                ):
                    q_val = q_values[y, x, i]
                    color = cmap(norm(q_val))
                    poly = Polygon(
                        tri, facecolor=color, edgecolor="black", linewidth=0.5
                    )
                    ax.add_patch(poly)

                    # Q値を表示
                    ax.text(
                        cx + offset[0],
                        cy + offset[1],
                        f"{q_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

                # 最適行動を中央に表示
                best_action = np.argmax(q_values[y, x])
                ax.text(
                    cx,
                    cy,
                    action_labels[best_action],
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="blue",
                )

    # スタートとゴールをマーク
    ax.text(
        0,
        0,
        "S",
        ha="center",
        va="center",
        fontsize=10,
        color="red",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="red"),
    )
    ax.text(
        goal[0],
        goal[1],
        "G",
        ha="center",
        va="center",
        fontsize=10,
        color="green",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="green"),
    )

    # グリッド線
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1)
        ax.axvline(i - 0.5, color="black", linewidth=1)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # y軸を反転
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Q-Values (each triangle = action direction)")

    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Q-value")

    plt.tight_layout()
    plt.show()


print("\nQ-Value Heatmap:")
plot_q_heatmap(agent)

# %% Evaluation
print("\nEvaluating trained agent...")
evaluate_agent(agent, num_episodes=3)
