# %% [markdown]
# # GridWorld DQN
# DQNを用いたGridWorldの学習

# %% Imports
import numpy as np
from collections import deque

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent_trainer import AgentTrainer, Env
from dpl.agent import ReplayBuffer, BaseAgent


# %% Feature Toggle
# 観測モード: "onehot" または "local"
# - onehot: one-hotエンコーディング (width * height次元)
# - local: 局所視野 (3x3の壁情報9次元 + ゴールへの正規化方向ベクトル2次元 = 11次元)
OBSERVATION_MODE = "local"

# フレームスタック数（過去K個の観測を連結して状態とする）
FRAME_STACK = 8


# %% Map Parser
def parse_grid_map(
    ascii_map: str,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], int, int]:
    """ASCII形式のマップ文字列をパースする

    Args:
        ascii_map: 改行区切りのASCII文字列
            S: スタート
            G: ゴール
            #: 障害物
            .: 通行可能

    Returns:
        (start, goal, obstacles, width, height)
    """
    lines = [line.strip() for line in ascii_map.strip().split("\n")]
    height = len(lines)

    start = None
    goal = None
    obstacles: set[tuple[int, int]] = set()

    for y, line in enumerate(lines):
        # スペース区切りでセルを分割
        cells = line.split()
        width = len(cells)

        for x, cell in enumerate(cells):
            if cell == "S":
                start = (x, y)
            elif cell == "G":
                goal = (x, y)
            elif cell == "#":
                obstacles.add((x, y))

    if start is None:
        raise ValueError("Start position 'S' not found in map")
    if goal is None:
        raise ValueError("Goal position 'G' not found in map")

    # 最初の行の幅を基準にする
    width = len(lines[0].split())

    return start, goal, obstacles, width, height


# テスト
MAP_ASCII = """
S . . . . . .
# # # # # . #
. . . . # . #
. # # . # . #
. # G . . . #
. # # # # # #
. . . . . . .
"""

start, goal, obstacles, width, height = parse_grid_map(MAP_ASCII)
print(f"Start: {start}")
print(f"Goal: {goal}")
print(f"Obstacles: {obstacles}")
print(f"Size: {width}x{height}")


# %% GridWorld Environment
class GridWorld(Env):
    """シンプルなGridWorld環境

    ASCIIマップからグリッドを生成し、エージェントがスタート地点からゴール地点まで移動する。
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

    def __init__(
        self,
        ascii_map: str = MAP_ASCII,
        max_steps: int = 200,
        random_start: bool = False,
        corner_start: bool = False,
        max_visit_count: int = 30,
    ):
        start, goal, obstacles, width, height = parse_grid_map(ascii_map)

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.max_visit_count = max_visit_count  # 同一セル訪問回数の閾値
        self.action_space_n = 4  # 上下左右
        self.random_start = random_start  # ランダムスタート
        self.corner_start = corner_start  # 四隅からスタート

        self.obstacles = obstacles
        self.goal = goal
        self.start = start

        self.reset()

    @property
    def state_size(self) -> int:
        """状態空間のサイズ"""
        if OBSERVATION_MODE == "onehot":
            return self.width * self.height
        else:  # local
            # 3x3の壁情報(9) + 方向ベクトル(2) = 11
            return 11

    def reset(self) -> np.ndarray:
        """環境をリセット

        スタート位置の優先順位: corner_start > random_start > 固定start
        """
        if self.corner_start:
            # 四隅から有効な位置を選択
            corners = [
                (0, 0),
                (self.width - 1, 0),
                (0, self.height - 1),
                (self.width - 1, self.height - 1),
            ]
            valid_corners = [
                c for c in corners if c not in self.obstacles and c != self.goal
            ]
            if valid_corners:
                start_pos = valid_corners[np.random.randint(len(valid_corners))]
            else:
                start_pos = self.start  # フォールバック
            self.agent_pos = list(start_pos)
        elif self.random_start:
            # 有効なスタート位置を収集
            valid_positions = []
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in self.obstacles and (x, y) != self.goal:
                        valid_positions.append((x, y))
            # ランダムに選択
            start_pos = valid_positions[np.random.randint(len(valid_positions))]
            self.agent_pos = list(start_pos)
        else:
            self.agent_pos = list(self.start)
        self.steps = 0
        self.visit_counts: dict[tuple[int, int], int] = {}  # 訪問回数カウンタ
        # 初期位置を訪問済みとしてカウント
        pos: tuple[int, int] = (self.agent_pos[0], self.agent_pos[1])
        self.visit_counts[pos] = 1
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """状態を返す"""
        if OBSERVATION_MODE == "onehot":
            return self._get_state_onehot()
        else:  # local
            return self._get_state_local()

    def _get_state_onehot(self) -> np.ndarray:
        """one-hotエンコーディング"""
        state = np.zeros(self.width * self.height, dtype=np.float32)
        idx = self.agent_pos[1] * self.width + self.agent_pos[0]
        state[idx] = 1.0
        return state

    def _get_state_local(self) -> np.ndarray:
        """局所視野: 3x3の壁情報 + ゴールへの正規化方向ベクトル"""
        x, y = self.agent_pos

        # 3x3の壁情報（障害物または範囲外なら1、通行可能なら0）
        local_view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                # 範囲外チェック
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    local_view.append(1.0)  # 範囲外は壁扱い
                elif (nx, ny) in self.obstacles:
                    local_view.append(1.0)  # 障害物
                else:
                    local_view.append(0.0)  # 通行可能

        # ゴールへの方向ベクトル（正規化）
        gx, gy = self.goal
        dir_x = gx - x
        dir_y = gy - y
        dist = np.sqrt(dir_x**2 + dir_y**2)
        if dist > 0:
            dir_x /= dist
            dir_y /= dist

        state = np.array(local_view + [dir_x, dir_y], dtype=np.float32)
        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """1ステップ実行

        Args:
            action: 行動（0:上, 1:下, 2:左, 3:右）

        Returns:
            observation: 新しい状態
            reward: 報酬
            terminated: ゴール到達でTrue
            truncated: 最大ステップ数超過 or 同一セル訪問回数超過でTrue
            info: 追加情報
        """
        self.steps += 1

        # 移動先を計算
        new_pos = self.agent_pos.copy()
        if action == 0:  # 上
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # 下
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 2:  # 左
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # 右
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        # 壁チェック（移動しなかった場合）
        hit_wall = new_pos == self.agent_pos

        # 障害物チェック
        if tuple(new_pos) in self.obstacles:
            # 障害物には移動できない
            hit_wall = True
        else:
            self.agent_pos = new_pos

        # 訪問回数をカウント
        pos: tuple[int, int] = (self.agent_pos[0], self.agent_pos[1])
        self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1

        # 報酬計算
        reward = -0.01  # ステップペナルティ
        if hit_wall:
            reward -= 0.1  # 壁ペナルティ

        # ゴール判定
        terminated = tuple(self.agent_pos) == self.goal
        if terminated:
            reward = 1.0

        # 最大ステップ超過 or 同一セル訪問回数超過
        truncated = (
            self.steps >= self.max_steps
            or self.visit_counts[pos] >= self.max_visit_count
        )

        return self._get_state(), reward, terminated, truncated, {}

    def render(self):
        """グリッドを表示"""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        # 障害物
        for ox, oy in self.obstacles:
            grid[oy][ox] = "#"

        # ゴール
        gx, gy = self.goal
        grid[gy][gx] = "G"

        # エージェント
        ax, ay = self.agent_pos
        grid[ay][ax] = "A"

        print("-" * (self.width * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.width * 2 + 1))


# %% FrameStackEnv Wrapper
class FrameStackEnv:
    """任意の環境をラップしてフレームスタックを追加するラッパー"""

    def __init__(self, env, k: int = 4):
        self.env = env
        self.k = k
        self.frames: deque = deque(maxlen=k)
        self._obs_dim: int | None = None

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        if self._obs_dim is None:
            self._obs_dim = obs.shape[0]

        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)

        return self._get_stacked()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        self.frames.append(obs)
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self) -> np.ndarray:
        """(k*obs_dim,) の1次元ベクトルにする"""
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)

    @property
    def state_size(self) -> int:
        if self._obs_dim is None:
            raise RuntimeError("Call reset() first to initialize obs dim.")
        return self._obs_dim * self.k

    def render(self):
        """ラップした環境のrenderを呼び出す"""
        self.env.render()


# %% Q-Network
class QNet(L.Sequential):
    """GridWorld用のQ-Network

    入力: 状態
        - onehot: one-hotエンコーディング (width * height次元)
        - local: 局所視野 (3x3壁情報9次元 + 正規化座標2次元 = 11次元)
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
def evaluate_agent(
    agent: DQNAgent,
    env: GridWorld | FrameStackEnv,
    start_positions: list[tuple[int, int]] | None = None,
):
    """学習済みエージェントを評価し、訪問したセルのポリシーを可視化

    Args:
        agent: 学習済みエージェント
        env: 環境（GridWorld or FrameStackEnv）
        start_positions: スタート位置のリスト。Noneの場合は環境のデフォルト動作
    """
    # ベース環境を取得
    base_env = env.env if isinstance(env, FrameStackEnv) else env

    num_episodes = len(start_positions) if start_positions else 3

    for episode in range(num_episodes):
        # スタート位置を設定
        if start_positions:
            start_pos = start_positions[episode]
            # 環境をリセットしてから手動で位置を設定
            env.reset()
            base_env.agent_pos = list(start_pos)
            base_env.steps = 0
            # FrameStackの場合はフレームバッファを再初期化
            if isinstance(env, FrameStackEnv):
                obs = base_env._get_state()
                env.frames.clear()
                for _ in range(env.k):
                    env.frames.append(obs)
                observation = env._get_stacked()
            else:
                observation = base_env._get_state()
        else:
            observation = env.reset()

        total_reward = 0.0
        steps = 0

        # 訪問記録: (x, y) -> action
        visited: dict[tuple[int, int], int] = {}

        start_x, start_y = base_env.agent_pos[0], base_env.agent_pos[1]
        print(
            f"\n=== Evaluation Episode {episode + 1} (start: ({start_x}, {start_y})) ==="
        )

        while True:
            # 現在位置を記録
            pos: tuple[int, int] = (base_env.agent_pos[0], base_env.agent_pos[1])

            state_var = Variable(observation.reshape(1, -1).astype(np.float32))
            q_values = agent.qnet(state_var)
            action = int(np.argmax(q_values.data_required[0]))

            # 訪問したセルにアクションを記録
            visited[pos] = action

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated:
                print(f"ゴール到達! Total Reward = {total_reward:.2f}, Steps = {steps}")
                break
            if truncated:
                print(f"時間切れ... Total Reward = {total_reward:.2f}, Steps = {steps}")
                break

        # 訪問したセルのポリシーを可視化
        _plot_visited_policy(base_env, visited, episode + 1, (start_x, start_y))


def _plot_visited_policy(
    env: GridWorld,
    visited: dict[tuple[int, int], int],
    episode: int,
    start_pos: tuple[int, int] | None = None,
):
    """訪問したセルのみにポリシー矢印を描画

    Args:
        env: GridWorld環境
        visited: 訪問したセル -> アクションの辞書
        episode: エピソード番号
        start_pos: スタート位置（Noneの場合はenv.startを使用）
    """
    from matplotlib.patches import Rectangle

    width = env.width
    height = env.height
    obstacles = env.obstacles
    goal = env.goal
    start = start_pos if start_pos is not None else env.start

    # 行動に対応する矢印のオフセット (dx, dy)
    arrow_directions = {
        0: (0, -0.3),  # 上
        1: (0, 0.3),  # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0),  # 右
    }

    _, ax = plt.subplots(figsize=(8, 8))

    # グリッドを描画
    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                color = "gray"
            elif (x, y) in visited:
                color = "lightgreen"  # 訪問したセル
            else:
                color = "white"

            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # 訪問したセルに矢印を描画
            if (x, y) in visited and (x, y) != goal:
                action = visited[(x, y)]
                dx, dy = arrow_directions[action]
                ax.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    head_width=0.15,
                    head_length=0.1,
                    fc="darkblue",
                    ec="darkblue",
                )

    # スタートとゴール
    ax.text(
        start[0],
        start[1],
        "S",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
        fontweight="bold",
    )
    ax.text(
        goal[0],
        goal[1],
        "G",
        ha="center",
        va="center",
        fontsize=12,
        color="green",
        fontweight="bold",
    )

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Episode {episode}: Start {start} → Goal (ε=0)")

    plt.tight_layout()
    plt.show()


# %% Training
print("Training DQN Agent on GridWorld...")

# ベース環境を作成
base_env = GridWorld(corner_start=True)  # トレーニングは四隅からスタート
base_eval_env = GridWorld(random_start=True)  # 評価はランダムスタート（汎化性能を確認）

# FrameStackでラップ（FRAME_STACK > 1 の場合）
if FRAME_STACK > 1:
    env = FrameStackEnv(base_env, k=FRAME_STACK)
    eval_env = FrameStackEnv(base_eval_env, k=FRAME_STACK)
    # state_sizeを取得するためにreset()を呼ぶ
    env.reset()
    eval_env.reset()
else:
    env = base_env
    eval_env = base_eval_env

agent = DQNAgent(state_size=env.state_size)

trainer = AgentTrainer(
    env=env,
    eval_env=eval_env,
    agent=agent,
    num_episodes=1000,
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


# %% Helper function for visualization
def _get_state_for_pos(x: int, y: int, env: GridWorld) -> np.ndarray:
    """指定位置の状態を取得（可視化用、フレームスタック対応）"""
    if OBSERVATION_MODE == "onehot":
        single_state = np.zeros(env.width * env.height, dtype=np.float32)
        single_state[y * env.width + x] = 1.0
    else:  # local
        # 3x3の壁情報
        local_view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
                    local_view.append(1.0)
                elif (nx, ny) in env.obstacles:
                    local_view.append(1.0)
                else:
                    local_view.append(0.0)
        # ゴールへの方向ベクトル（正規化）
        gx, gy = env.goal
        dir_x = gx - x
        dir_y = gy - y
        dist = np.sqrt(dir_x**2 + dir_y**2)
        if dist > 0:
            dir_x /= dist
            dir_y /= dist
        single_state = np.array(local_view + [dir_x, dir_y], dtype=np.float32)

    # フレームスタック: 同じ状態をK回繰り返す（静止状態を想定）
    if FRAME_STACK > 1:
        return np.tile(single_state, FRAME_STACK)
    return single_state


# %% Q-Value Heatmap
def plot_q_heatmap(agent: DQNAgent, env: GridWorld):
    """各セルのQ値を4方向の三角形で可視化

    各セルを対角線で4つの三角形に分割:
    - 上: 上方向への移動のQ値
    - 下: 下方向への移動のQ値
    - 左: 左方向への移動のQ値
    - 右: 右方向への移動のQ値
    """
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors

    width = env.width
    height = env.height
    obstacles = env.obstacles
    goal = env.goal
    start = env.start

    # 各セルのQ値を計算
    q_values = np.zeros((height, width, 4))

    for y in range(height):
        for x in range(width):
            state = _get_state_for_pos(x, y, env)
            state_var = Variable(state.reshape(1, -1))
            q = agent.qnet(state_var).data_required[0]
            q_values[y, x] = q

    # Q値の範囲を取得（カラーマップ用）
    q_min = np.min(q_values)
    q_max = np.max(q_values)
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.get_cmap("RdYlGn")

    _, ax = plt.subplots(figsize=(12, 10))

    # 各セルに4つの三角形を描画
    for y in range(height):
        for x in range(width):
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
                        fontsize=6,
                        color="black",
                    )

    # スタートとゴールをマーク
    ax.text(
        start[0],
        start[1],
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
    for i in range(height + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1)
    for i in range(width + 1):
        ax.axvline(i - 0.5, color="black", linewidth=1)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # y軸を反転
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
plot_q_heatmap(agent, base_env)


# %% Value Heatmap and Policy
def plot_value_and_policy(agent: DQNAgent, env: GridWorld):
    """Value関数のheatmapとPolicyの矢印を並べて表示

    左: V(s) = max_a Q(s, a) のheatmap
    右: 各セルでの最適行動を矢印で表示
    """
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    width = env.width
    height = env.height
    obstacles = env.obstacles
    goal = env.goal
    start = env.start

    # 各セルのQ値を計算
    q_values = np.zeros((height, width, 4))

    for y in range(height):
        for x in range(width):
            state = _get_state_for_pos(x, y, env)
            state_var = Variable(state.reshape(1, -1))
            q = agent.qnet(state_var).data_required[0]
            q_values[y, x] = q

    # V(s) = max_a Q(s, a)
    v_values = np.max(q_values, axis=2)

    # 最適行動
    best_actions = np.argmax(q_values, axis=2)

    # 行動に対応する矢印のオフセット (dx, dy)
    # action: 0=上, 1=下, 2=左, 3=右
    arrow_directions = {
        0: (0, -0.3),  # 上
        1: (0, 0.3),  # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0),  # 右
    }

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # === 左: Value Heatmap ===
    # 障害物を除いたV値の範囲
    v_min = np.min(v_values)
    v_max = np.max(v_values)
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap("RdYlGn")

    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                color = "gray"
            else:
                color = cmap(norm(v_values[y, x]))

            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax1.add_patch(rect)

            if (x, y) not in obstacles:
                ax1.text(
                    x,
                    y,
                    f"{v_values[y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    # スタートとゴール
    ax1.text(
        start[0],
        start[1],
        "S",
        ha="center",
        va="center",
        fontsize=10,
        color="red",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="red"),
    )
    ax1.text(
        goal[0],
        goal[1],
        "G",
        ha="center",
        va="center",
        fontsize=10,
        color="green",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="green"),
    )

    ax1.set_xlim(-0.5, width - 0.5)
    ax1.set_ylim(height - 0.5, -0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Value Function V(s) = max_a Q(s, a)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="V(s)")

    # === 右: Policy Arrows ===
    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                color = "gray"
            else:
                color = "lightblue"

            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax2.add_patch(rect)

            if (x, y) not in obstacles and (x, y) != goal:
                action = best_actions[y, x]
                dx, dy = arrow_directions[action]
                ax2.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    head_width=0.15,
                    head_length=0.1,
                    fc="darkblue",
                    ec="darkblue",
                )

    # スタートとゴール
    ax2.text(
        start[0],
        start[1],
        "S",
        ha="center",
        va="center",
        fontsize=10,
        color="red",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="red"),
    )
    ax2.text(
        goal[0],
        goal[1],
        "G",
        ha="center",
        va="center",
        fontsize=10,
        color="green",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="green"),
    )

    ax2.set_xlim(-0.5, width - 0.5)
    ax2.set_ylim(height - 0.5, -0.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Policy π(s) = argmax_a Q(s, a)")

    plt.tight_layout()
    plt.show()


print("\nValue and Policy:")
plot_value_and_policy(agent, base_env)


# %% Evaluation
print("\nEvaluating trained agent...")


# スタート位置を生成: y=0から2つ、y=6から2つ、その他から2つ
def generate_start_positions(env: GridWorld) -> list[tuple[int, int]]:
    """評価用のスタート位置を生成"""
    obstacles = env.obstacles
    goal = env.goal

    # 各行ごとに有効な位置を収集
    row_0_positions = [
        (x, 0) for x in range(env.width) if (x, 0) not in obstacles and (x, 0) != goal
    ]
    row_6_positions = [
        (x, 6) for x in range(env.width) if (x, 6) not in obstacles and (x, 6) != goal
    ]
    other_positions = [
        (x, y)
        for y in range(1, 6)  # y=1~5
        for x in range(env.width)
        if (x, y) not in obstacles and (x, y) != goal
    ]

    # ランダムに選択
    np.random.seed(42)  # 再現性のため
    positions = []
    positions.extend(
        [
            row_0_positions[i]
            for i in np.random.choice(len(row_0_positions), 2, replace=False)
        ]
    )
    positions.extend(
        [
            row_6_positions[i]
            for i in np.random.choice(len(row_6_positions), 2, replace=False)
        ]
    )
    positions.extend(
        [
            other_positions[i]
            for i in np.random.choice(len(other_positions), 2, replace=False)
        ]
    )

    return positions


start_positions = generate_start_positions(base_env)
print(f"Start positions: {start_positions}")
evaluate_agent(agent, eval_env, start_positions=start_positions)
