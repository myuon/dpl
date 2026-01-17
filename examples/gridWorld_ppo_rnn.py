# %% [markdown]
# # GridWorld PPO+RNN
# PPO + Recurrent Network (LSTM) を用いたGridWorldの学習
#
# gridWorld_dqn.py と同じ部分観測環境（3x3視野のみ）で学習

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from models.ppo_rnn import (
    RecurrentActorCritic,
    PPORNNAgent,
)


# %% Map Parser (from gridWorld_dqn.py)
def parse_grid_map(
    ascii_map: str,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], int, int]:
    """ASCII形式のマップ文字列をパース"""
    lines = [line.strip() for line in ascii_map.strip().split("\n")]
    height = len(lines)

    start = None
    goal = None
    obstacles: set[tuple[int, int]] = set()

    for y, line in enumerate(lines):
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

    width = len(lines[0].split())
    return start, goal, obstacles, width, height


def rotate_map(ascii_map: str, times: int = 1) -> str:
    """マップを時計回りに90度×times回転させる"""
    lines = [line.strip() for line in ascii_map.strip().split("\n")]
    grid = [line.split() for line in lines]

    for _ in range(times % 4):
        height = len(grid)
        width = len(grid[0])
        rotated = [
            [grid[height - 1 - j][i] for j in range(height)] for i in range(width)
        ]
        grid = rotated

    return "\n".join(" ".join(row) for row in grid)


MAP_ASCII = """
S . . . . . .
# # # # # . #
. . . . # . #
. # # . # . #
. # G . . . #
. # # # # # #
. . . . . . .
"""

# 観測モード: 3x3の壁情報のみ（部分観測）
OBSERVATION_MODE = "local_partial"


# %% GridWorld Environment (from gridWorld_dqn.py)
class GridWorld:
    """シンプルなGridWorld環境（部分観測モード）"""

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
        self.max_visit_count = max_visit_count
        self.action_space_n = 4
        self.random_start = random_start
        self.corner_start = corner_start

        self.obstacles = obstacles
        self.goal = goal
        self.start = start

        self.reset()

    @property
    def state_size(self) -> int:
        """状態空間のサイズ（3x3の壁情報 = 9次元）"""
        return 9

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """環境をリセット"""
        if seed is not None:
            np.random.seed(seed)
        if self.corner_start:
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
                start_pos = self.start
            self.agent_pos = list(start_pos)
        elif self.random_start:
            valid_positions = []
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in self.obstacles and (x, y) != self.goal:
                        valid_positions.append((x, y))
            start_pos = valid_positions[np.random.randint(len(valid_positions))]
            self.agent_pos = list(start_pos)
        else:
            self.agent_pos = list(self.start)

        self.steps = 0
        self.visit_counts: dict[tuple[int, int], int] = {}
        pos: tuple[int, int] = (self.agent_pos[0], self.agent_pos[1])
        self.visit_counts[pos] = 1
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """部分観測: 3x3の壁情報のみ"""
        x, y = self.agent_pos
        local_view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    local_view.append(1.0)  # 範囲外は壁扱い
                elif (nx, ny) in self.obstacles:
                    local_view.append(1.0)  # 障害物
                else:
                    local_view.append(0.0)  # 通行可能
        return np.array(local_view, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """1ステップ実行"""
        self.steps += 1

        new_pos = self.agent_pos.copy()
        if action == 0:  # 上
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # 下
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 2:  # 左
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # 右
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        hit_wall = new_pos == self.agent_pos

        if tuple(new_pos) in self.obstacles:
            hit_wall = True
        else:
            self.agent_pos = new_pos

        pos: tuple[int, int] = (self.agent_pos[0], self.agent_pos[1])
        self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1

        reward = -0.01
        if hit_wall:
            reward -= 0.1

        terminated = tuple(self.agent_pos) == self.goal
        if terminated:
            reward = 1.0

        # local_partialモードではループ検知を無効化
        truncated = self.steps >= self.max_steps

        if truncated and not terminated:
            reward -= 1.0

        return self._get_state(), reward, terminated, truncated, {}

    def render(self):
        """グリッドを表示"""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        for ox, oy in self.obstacles:
            grid[oy][ox] = "#"

        gx, gy = self.goal
        grid[gy][gx] = "G"

        ax, ay = self.agent_pos
        grid[ay][ax] = "A"

        print("-" * (self.width * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.width * 2 + 1))


class RotatedMapGridWorld(GridWorld):
    """リセット時にマップを回転させるGridWorld"""

    def __init__(
        self,
        base_map: str = MAP_ASCII,
        max_steps: int = 200,
        max_visit_count: int = 30,
    ):
        self._base_map = base_map
        self._rotations = [1, 2, 3]  # 90度, 180度, 270度

        initial_map = rotate_map(base_map, self._rotations[0])
        super().__init__(
            ascii_map=initial_map,
            max_steps=max_steps,
            random_start=False,
            corner_start=True,
            max_visit_count=max_visit_count,
        )

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """リセット時にランダムな回転を適用"""
        if seed is not None:
            np.random.seed(seed)
        rotation = self._rotations[np.random.randint(len(self._rotations))]
        rotated_map = rotate_map(self._base_map, rotation)
        start, goal, obstacles, width, height = parse_grid_map(rotated_map)

        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.goal = goal
        self.start = start

        return super().reset()


# %% VectorEnvWrapper for single env
class SingleEnvWrapper:
    """単一環境をVectorEnv形式でラップ

    PPORNNAgentのcollect_rollout()で使用するため
    """

    def __init__(self, env_fn):
        """
        Args:
            env_fn: 環境を生成する関数
        """
        self.env = env_fn()
        self.n_envs = 1
        self._last_obs = None

    @property
    def state_size(self):
        return self.env.state_size

    def reset(self) -> np.ndarray:
        """環境をリセット

        Returns:
            obs: (n_envs, obs_dim) = (1, obs_dim)
        """
        obs = self.env.reset()
        self._last_obs = obs
        return obs[None, :]  # (1, obs_dim)

    def reset_single(self, env_id: int) -> np.ndarray:
        """指定環境をリセット

        Returns:
            obs: (obs_dim,)
        """
        obs = self.env.reset()
        self._last_obs = obs
        return obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """1ステップ実行

        Args:
            actions: (n_envs,) = (1,)

        Returns:
            obs: (n_envs, obs_dim)
            rewards: (n_envs,)
            terminateds: (n_envs,)
            truncateds: (n_envs,)
        """
        action = int(actions[0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        return (
            obs[None, :],
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=bool),
            np.array([truncated], dtype=bool),
        )


class MultiEnvWrapper:
    """複数環境を並列実行するラッパー"""

    def __init__(self, env_fn, n_envs: int):
        """
        Args:
            env_fn: 環境を生成する関数
            n_envs: 並列環境数
        """
        self.envs = [env_fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        self._last_obs = None

    @property
    def state_size(self):
        return self.envs[0].state_size

    def reset(self) -> np.ndarray:
        """全環境をリセット

        Returns:
            obs: (n_envs, obs_dim)
        """
        obs = np.array([env.reset() for env in self.envs], dtype=np.float32)
        self._last_obs = obs
        return obs

    def reset_single(self, env_id: int) -> np.ndarray:
        """指定環境をリセット

        Returns:
            obs: (obs_dim,)
        """
        obs = self.envs[env_id].reset()
        return obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """全環境で1ステップ実行

        Args:
            actions: (n_envs,)

        Returns:
            obs: (n_envs, obs_dim)
            rewards: (n_envs,)
            terminateds: (n_envs,)
            truncateds: (n_envs,)
        """
        results = [env.step(int(a)) for env, a in zip(self.envs, actions)]

        obs = np.array([r[0] for r in results], dtype=np.float32)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminateds = np.array([r[2] for r in results], dtype=bool)
        truncateds = np.array([r[3] for r in results], dtype=bool)

        return obs, rewards, terminateds, truncateds


# %% Evaluation function
def evaluate_ppo_rnn(
    agent: PPORNNAgent,
    env: GridWorld,
    n_episodes: int = 20,
    render_episodes: int = 0,
) -> tuple[float, float, float]:
    """PPO+RNNエージェントを評価

    Args:
        agent: 評価対象のエージェント
        env: 評価環境
        n_episodes: 評価エピソード数
        render_episodes: 可視化するエピソード数（最初のn個）

    Returns:
        avg_return: 平均リターン
        success_rate: 成功率
        avg_steps: 平均ステップ数（成功時のみ）
    """
    returns = []
    steps_to_goal = []

    for ep in range(n_episodes):
        obs = env.reset()
        agent.reset_state()  # LSTM状態をリセット

        total_reward = 0.0
        steps = 0
        trajectory = []  # 可視化用

        while True:
            # 行動選択（explore=Falseでもstochastic）
            action = agent.act(obs, explore=False)
            trajectory.append((env.agent_pos.copy(), action))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        returns.append(total_reward)
        if terminated:
            steps_to_goal.append(steps)

        # 可視化
        if ep < render_episodes:
            _plot_trajectory(env, trajectory, ep + 1, terminated)

    avg_return = float(np.mean(returns))
    success_rate = len(steps_to_goal) / n_episodes
    avg_steps = float(np.mean(steps_to_goal)) if steps_to_goal else 0.0

    return avg_return, success_rate, avg_steps


def _plot_trajectory(env: GridWorld, trajectory: list, episode: int, success: bool):
    """エピソードの軌跡を可視化"""
    from matplotlib.patches import Rectangle

    width = env.width
    height = env.height
    obstacles = env.obstacles
    goal = env.goal
    start = trajectory[0][0] if trajectory else env.start

    # 行動に対応する矢印のオフセット (dx, dy)
    arrow_directions = {
        0: (0, -0.3),  # 上
        1: (0, 0.3),   # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0),   # 右
    }

    _, ax = plt.subplots(figsize=(8, 8))

    # 訪問したセルを記録
    visited = {tuple(pos): action for pos, action in trajectory}

    # グリッドを描画
    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                color = "gray"
            elif (x, y) in visited:
                color = "lightgreen"
            else:
                color = "white"

            rect = Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # 訪問したセルに矢印を描画
            if (x, y) in visited and (x, y) != tuple(goal):
                action = visited[(x, y)]
                dx, dy = arrow_directions[action]
                ax.arrow(x, y, dx, dy,
                         head_width=0.15, head_length=0.1,
                         fc="darkblue", ec="darkblue")

    # スタートとゴール
    ax.text(start[0], start[1], "S", ha="center", va="center",
            fontsize=12, color="red", fontweight="bold")
    ax.text(goal[0], goal[1], "G", ha="center", va="center",
            fontsize=12, color="green", fontweight="bold")

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    result_str = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Episode {episode}: {result_str}")

    plt.tight_layout()
    plt.show()


# %% Debug PPORNNAgent (collect_rolloutにデバッグ統計を追加)
class DebugPPORNNAgent(PPORNNAgent):
    """デバッグ統計付きPPORNNAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # デバッグ統計
        self.debug_n_terminated: int = 0
        self.debug_n_truncated: int = 0
        self.debug_goal_reward_count: int = 0
        self.debug_adv_mean: float = 0.0
        self.debug_adv_std: float = 0.0
        self.debug_adv_min: float = 0.0
        self.debug_adv_max: float = 0.0
        self.debug_ret_mean: float = 0.0
        self.debug_ret_std: float = 0.0
        self.debug_ret_min: float = 0.0
        self.debug_ret_max: float = 0.0

    def collect_rollout(self, env) -> tuple[int, list[float]]:
        """デバッグ統計付きcollect_rollout"""
        # バッファをクリア
        self.buffer.clear()

        obs = env.reset()  # (n_envs, obs_dim)
        self.reset_state()

        episode_rewards: list[float] = []
        current_rewards = np.zeros(self.n_envs)
        episode_starts = np.ones(self.n_envs, dtype=np.float32)

        # デバッグカウンタ
        n_terminated = 0
        n_truncated = 0
        goal_reward_count = 0

        for _ in range(self.rollout_len):
            actions = self.get_actions(obs)
            next_obs, rewards, terminateds, truncateds = env.step(actions)
            dones = terminateds | truncateds

            # ゴール報酬カウント
            goal_reward_count += int(np.sum(rewards >= 0.9))

            # 保存
            self.buffer.add(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones.astype(np.float32),
                log_probs=self._last_log_probs,
                values=self._last_values,
                episode_starts=episode_starts,
            )

            current_rewards += rewards

            # done環境の処理
            done_mask = dones.astype(bool)
            if np.any(done_mask):
                for i in np.where(done_mask)[0]:
                    episode_rewards.append(current_rewards[i])
                    current_rewards[i] = 0.0
                    if terminateds[i]:
                        n_terminated += 1
                    if truncateds[i]:
                        n_truncated += 1
                    next_obs[i] = env.reset_single(i)
                    self.reset_state(env_id=i)

            obs = next_obs
            episode_starts = dones.astype(np.float32)

        # 最終観測からV(s)を計算
        _, last_values = self.actor_critic.steps(obs)

        # GAE計算
        self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

        # GAE統計を保存
        adv = self.buffer.advantages[:self.buffer.ptr]
        ret = self.buffer.returns[:self.buffer.ptr]

        self.debug_n_terminated = n_terminated
        self.debug_n_truncated = n_truncated
        self.debug_goal_reward_count = goal_reward_count
        self.debug_adv_mean = float(np.mean(adv))
        self.debug_adv_std = float(np.std(adv))
        self.debug_adv_min = float(np.min(adv))
        self.debug_adv_max = float(np.max(adv))
        self.debug_ret_mean = float(np.mean(ret))
        self.debug_ret_std = float(np.std(ret))
        self.debug_ret_min = float(np.min(ret))
        self.debug_ret_max = float(np.max(ret))

        return self.n_envs * self.rollout_len, episode_rewards


def debug_stats_extractor(agent) -> str | None:
    """デバッグ統計を出力するstats_extractor"""
    parts = []

    # PPO基本統計
    if agent.last_policy_loss is not None:
        parts.append(f"π={agent.last_policy_loss:.4f}")
    if agent.last_value_loss is not None:
        parts.append(f"V={agent.last_value_loss:.4f}")
    if agent.last_entropy is not None:
        parts.append(f"H={agent.last_entropy:.3f}")

    # Rollout統計
    if hasattr(agent, "debug_n_terminated"):
        parts.append(f"term={agent.debug_n_terminated}")
        parts.append(f"trunc={agent.debug_n_truncated}")
        parts.append(f"goal={agent.debug_goal_reward_count}")

    # GAE統計
    if hasattr(agent, "debug_adv_mean"):
        parts.append(f"adv={agent.debug_adv_mean:.3f}±{agent.debug_adv_std:.3f}")
        parts.append(f"ret={agent.debug_ret_mean:.3f}±{agent.debug_ret_std:.3f}")

    if not parts:
        return None
    return ", " + ", ".join(parts)


# %% Training configuration
print(f"Observation mode: {OBSERVATION_MODE}")
print(f"Map:\n{MAP_ASCII.strip()}")

# 環境設定
N_ENVS = 4  # 並列環境数
ROLLOUT_LEN = 128  # rolloutの長さ
NUM_UPDATES = 300  # 更新回数
EVAL_INTERVAL = 20  # 評価間隔

# 環境を作成
def make_env():
    return GridWorld(corner_start=True, max_steps=200)

def make_eval_env():
    return RotatedMapGridWorld(max_steps=200)

# ベクトル環境
if N_ENVS == 1:
    vec_env = SingleEnvWrapper(make_env)
else:
    vec_env = MultiEnvWrapper(make_env, n_envs=N_ENVS)

obs_dim = vec_env.state_size
action_dim = 4  # 上下左右

print(f"\nEnvironment: GridWorld (local_partial)")
print(f"  obs_dim: {obs_dim}")
print(f"  action_dim: {action_dim}")
print(f"  n_envs: {N_ENVS}")
print(f"  rollout_len: {ROLLOUT_LEN}")


# %% Create agent
actor_critic = RecurrentActorCritic(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_size=64,
    n_envs=N_ENVS,
)

agent = DebugPPORNNAgent(
    actor_critic=actor_critic,
    action_size=action_dim,
    obs_dim=obs_dim,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.02,  # 探索促進
    value_coef=0.5,
    lr=3e-4,
    n_epochs=4,
    seq_len=16,
    burn_in=4,
    batch_size=16,
    rollout_steps=N_ENVS * ROLLOUT_LEN,
    max_grad_norm=0.5,
    target_kl=0.02,
    n_envs=N_ENVS,
    rollout_len=ROLLOUT_LEN,
)

print("\nAgent: PPO+RNN (Debug)")
print(f"  hidden_size: 64")
print(f"  seq_len: 16")
print(f"  burn_in: 4")
print(f"  batch_size: 16")


# %% Training with ParallelAgentTrainer
from dpl.agent_trainer import ParallelAgentTrainer

print("\n" + "=" * 50)
print("Training PPO+RNN on GridWorld...")
print("=" * 50)

# 評価用環境（単一）
eval_env = make_eval_env()

# 評価用エージェント（n_envs=1）
eval_actor_critic = RecurrentActorCritic(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_size=64,
    n_envs=1,
)
eval_agent = DebugPPORNNAgent(
    actor_critic=eval_actor_critic,
    action_size=action_dim,
    obs_dim=obs_dim,
    n_envs=1,
    rollout_len=1,
)


def get_weights():
    """学習エージェントから重みを取得"""
    return [p.data.copy() if p.data is not None else None for p in actor_critic.params()]


def set_weights(agent, weights):
    """評価エージェントに重みを設定"""
    for p, w in zip(agent.actor_critic.params(), weights):
        if w is not None:
            p.data = w.copy()


trainer = ParallelAgentTrainer(
    env=vec_env,
    agent=agent,
    eval_env=eval_env,
    eval_agent=eval_agent,
    get_weights=get_weights,
    set_weights=set_weights,
    num_updates=NUM_UPDATES,
    eval_interval=EVAL_INTERVAL,
    eval_n=50,
    log_interval=10,
    stats_extractor=debug_stats_extractor,
)

result = trainer.train()


# %% Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# 1. Episode Rewards (移動平均)
window = 50
all_episode_rewards = result.episode_rewards
if len(all_episode_rewards) >= window:
    smoothed = np.convolve(all_episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, alpha=0.8)
else:
    ax1.plot(all_episode_rewards, alpha=0.7)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title(f"Episode Rewards (moving avg, window={window})")
ax1.grid(True)

# 2. Eval Return
if result.eval_returns:
    updates, returns = zip(*result.eval_returns)
    ax2.plot(updates, returns, marker='o', markersize=4, linewidth=1.5)
ax2.set_xlabel("Update")
ax2.set_ylabel("Eval Return")
ax2.set_title("Eval Return")
ax2.grid(True)

# 3. Success Rate
if result.eval_success_rates:
    updates, rates = zip(*result.eval_success_rates)
    ax3.plot(updates, [r*100 for r in rates], marker='o', markersize=4,
             linewidth=1.5, color='green')
ax3.set_xlabel("Update")
ax3.set_ylabel("Success Rate (%)")
ax3.set_title("Success Rate")
ax3.set_ylim(0, 105)
ax3.grid(True)

# 4. 未使用（ParallelTrainResultにはeval_avg_stepsがない）
ax4.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
ax4.set_title("(reserved)")

plt.tight_layout()
plt.show()


# %% Final evaluation with visualization
print("\n" + "=" * 50)
print("Final Evaluation (with trajectory visualization)")
print("=" * 50)

# 重みをコピー
set_weights(eval_agent, get_weights())

# 最終評価（軌跡を表示）
avg_ret, succ_rate, avg_st = evaluate_ppo_rnn(
    eval_agent, eval_env, n_episodes=10, render_episodes=6
)

print(f"\nFinal Results:")
print(f"  Average Return: {avg_ret:.2f}")
print(f"  Success Rate: {succ_rate*100:.0f}%")
print(f"  Avg Steps to Goal: {avg_st:.1f}")
