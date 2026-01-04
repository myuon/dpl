from collections import defaultdict

import numpy as np


class GridWorld:
    """シンプルなグリッドワールド環境"""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self):
        self.height = 3
        self.width = 4
        self.start = (2, 0)  # 左下
        self.state = self.start

        # 報酬マップ: 右上(0,3)が+1、その1マス下(1,3)が-1
        self.reward_map = {
            (0, 3): +1.0,
            (1, 3): -1.0,
        }

        # ゴール（エピソード終了）
        self.goal = (0, 3)

        # 壁（侵入不可）
        self.walls = {(1, 1)}

    def reset(self) -> tuple[int, int]:
        """環境をリセットして初期状態を返す"""
        self.state = self.start
        return self.state

    def step(self, action: int) -> tuple[tuple[int, int], float | None, bool]:
        """アクションを実行して次の状態、報酬、終了フラグを返す"""
        y, x = self.state

        if action == self.UP:
            y = max(0, y - 1)
        elif action == self.DOWN:
            y = min(self.height - 1, y + 1)
        elif action == self.LEFT:
            x = max(0, x - 1)
        elif action == self.RIGHT:
            x = min(self.width - 1, x + 1)

        # 壁には進めない
        if (y, x) in self.walls:
            y, x = self.state

        self.state = (y, x)
        reward = self.reward_map.get(self.state, None)
        done = self.state == self.goal  # ゴールに到達したら終了

        return self.state, reward, done

    def get_actions(self) -> list[int]:
        """利用可能なアクションのリストを返す"""
        return [self.UP, self.DOWN, self.LEFT, self.RIGHT]

    def states(self):
        """すべての状態(y, x)を返す"""
        for y in range(self.height):
            for x in range(self.width):
                yield (y, x)

    def render(self) -> str:
        """現在の状態を文字列で表示"""
        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (y, x) == self.state:
                    row.append("A")
                elif (y, x) in self.walls:
                    row.append("#")
                elif (y, x) in self.reward_map:
                    r = self.reward_map[(y, x)]
                    row.append("+" if r > 0 else "-")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        return "\n".join(grid)

    def render_v_pi(self, V, pi):
        """価値関数と方策を可視化"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))

        # 価値関数を2D配列に変換
        v_grid = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                v_grid[y, x] = V[(y, x)]

        im = ax.imshow(v_grid, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))

        # 矢印の方向
        arrow_map = {
            self.UP: (0, -0.3),
            self.DOWN: (0, 0.3),
            self.LEFT: (-0.3, 0),
            self.RIGHT: (0.3, 0),
        }

        # 各セルに値と矢印を表示
        for y in range(self.height):
            for x in range(self.width):
                state = (y, x)
                if state in self.walls:
                    ax.text(
                        x, y, "#", ha="center", va="center", fontsize=14, color="black"
                    )
                else:
                    # 報酬マスの場合は右上に報酬を表示
                    if state in self.reward_map:
                        ax.text(
                            x + 0.3,
                            y - 0.3,
                            f"{self.reward_map[state]:+.1f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="blue",
                        )
                    # 価値を表示
                    ax.text(
                        x,
                        y + 0.25,
                        f"{V[state]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )
                    # 矢印を表示（確率に応じた透明度）
                    if state in pi:
                        for action, prob in pi[state].items():
                            if prob > 0:
                                dx, dy = arrow_map[action]
                                scale = prob  # 確率に応じてスケール
                                ax.arrow(
                                    x - dx * scale / 2,
                                    y - 0.1 - dy * scale / 2,
                                    dx * scale,
                                    dy * scale,
                                    head_width=0.12 * scale,
                                    head_length=0.08 * scale,
                                    fc="black",
                                    ec="black",
                                    alpha=prob,
                                )

        fig.colorbar(im, ax=ax)
        ax.set_title("Value Function & Policy")
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        plt.show()

    def render_q(self, Q: dict[tuple[tuple[int, int], int], float]):
        """Q(s,a)を可視化: 各セルを4方向に分割してQ値を表示"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=(8, 6))

        # カラーマップの範囲を決定
        q_values = [
            Q[(s, a)]
            for s in self.states()
            if s not in self.walls
            for a in self.get_actions()
        ]
        if q_values:
            vmin, vmax = min(q_values), max(q_values)
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
        else:
            vmin, vmax = -1, 1

        cmap = plt.get_cmap("RdYlGn")
        norm = Normalize(vmin=vmin, vmax=vmax)

        for y in range(self.height):
            for x in range(self.width):
                state = (y, x)
                if state in self.walls:
                    # 壁は灰色で塗りつぶし
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, color="gray")
                    ax.add_patch(rect)
                    ax.text(
                        x, y, "#", ha="center", va="center", fontsize=12, color="white"
                    )
                else:
                    # セルの中心と4隅の座標
                    cx, cy = x, y
                    corners = {
                        "top_left": (cx - 0.5, cy - 0.5),
                        "top_right": (cx + 0.5, cy - 0.5),
                        "bottom_left": (cx - 0.5, cy + 0.5),
                        "bottom_right": (cx + 0.5, cy + 0.5),
                    }

                    # 各方向の三角形を描画
                    triangles = {
                        self.UP: [corners["top_left"], corners["top_right"], (cx, cy)],
                        self.DOWN: [
                            corners["bottom_right"],
                            corners["bottom_left"],
                            (cx, cy),
                        ],
                        self.LEFT: [
                            corners["top_left"],
                            corners["bottom_left"],
                            (cx, cy),
                        ],
                        self.RIGHT: [
                            corners["top_right"],
                            corners["bottom_right"],
                            (cx, cy),
                        ],
                    }

                    # テキスト位置
                    text_pos = {
                        self.UP: (cx, cy - 0.25),
                        self.DOWN: (cx, cy + 0.25),
                        self.LEFT: (cx - 0.25, cy),
                        self.RIGHT: (cx + 0.25, cy),
                    }

                    for action in self.get_actions():
                        q_val = Q[(state, action)]
                        color = cmap(norm(q_val))
                        triangle = Polygon(
                            triangles[action],
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        ax.add_patch(triangle)

                        # Q値を表示
                        tx, ty = text_pos[action]
                        ax.text(
                            tx,
                            ty,
                            f"{q_val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black",
                        )

        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_title("Q(s, a) Values")
        ax.set_aspect("equal")
        plt.show()


class RandomGenGridWorld(GridWorld):
    """ランダム生成のグリッドワールド環境"""

    def __init__(self, seed: int | None = None):
        # 親クラスの__init__は呼ばずに独自に初期化
        self.height = 10
        self.width = 10
        self.start = (9, 0)  # 左下
        self.goal = (0, 9)  # 右上
        self.state = self.start

        rng = np.random.default_rng(seed)

        # 利用可能なセル（start, goal以外）
        available = [
            (y, x)
            for y in range(self.height)
            for x in range(self.width)
            if (y, x) not in {self.start, self.goal}
        ]
        rng.shuffle(available)

        total_cells = len(available)
        n_walls = int(total_cells * 0.2)
        n_rewards = int(total_cells * 0.2)

        # 壁を配置
        self.walls = set(available[:n_walls])

        # 報酬を配置（壁以外の場所から選択）
        reward_candidates = available[n_walls : n_walls + n_rewards]
        self.reward_map = {pos: rng.choice([-1.0, 1.0]) for pos in reward_candidates}

        # ゴールには+100報酬
        self.reward_map[self.goal] = 100.0

    def render_v_pi(self, V, pi):
        """価値関数と方策を可視化（大きなグリッド用）"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 12))

        # 価値関数を2D配列に変換
        v_grid = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                v_grid[y, x] = V[(y, x)]

        im = ax.imshow(v_grid, cmap="RdYlGn", vmin=-10, vmax=100)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))

        # 矢印の方向
        arrow_map = {
            self.UP: (0, -0.3),
            self.DOWN: (0, 0.3),
            self.LEFT: (-0.3, 0),
            self.RIGHT: (0.3, 0),
        }

        # startからgoalまでの最適パスを計算
        optimal_path = set()  # (state, action) のセット
        current = self.start
        visited = set()
        max_steps = self.height * self.width  # 無限ループ防止
        for _ in range(max_steps):
            if current == self.goal or current in visited:
                break
            visited.add(current)
            if current in pi:
                # max actionを選択
                best_action = max(pi[current], key=lambda a: pi[current][a])
                optimal_path.add((current, best_action))
                # 次の状態へ移動
                self.state = current
                next_state, _, _ = self.step(best_action)
                current = next_state

        # 各セルに値と矢印を表示
        for y in range(self.height):
            for x in range(self.width):
                state = (y, x)
                if state in self.walls:
                    ax.text(
                        x, y, "#", ha="center", va="center", fontsize=10, color="black"
                    )
                else:
                    # 報酬マスの場合は右上に報酬を表示
                    if state in self.reward_map:
                        ax.text(
                            x + 0.3,
                            y - 0.3,
                            f"{self.reward_map[state]:+.0f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="blue",
                        )
                    # 価値を表示
                    ax.text(
                        x,
                        y + 0.25,
                        f"{V[state]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )
                    # 矢印を表示（確率に応じた透明度）
                    if state in pi:
                        for action, prob in pi[state].items():
                            if prob > 0:
                                dx, dy = arrow_map[action]
                                scale = prob * 0.8
                                # 最適パス上の矢印は赤色
                                is_on_path = (state, action) in optimal_path
                                color = "red" if is_on_path else "black"
                                ax.arrow(
                                    x - dx * scale / 2,
                                    y - 0.1 - dy * scale / 2,
                                    dx * scale,
                                    dy * scale,
                                    head_width=0.1 * scale,
                                    head_length=0.06 * scale,
                                    fc=color,
                                    ec=color,
                                    alpha=1.0 if is_on_path else prob,
                                )

        fig.colorbar(im, ax=ax)
        ax.set_title("Value Function & Policy (RandomGenGridWorld)")
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        plt.show()


# =============================================================================
# モンテカルロ法 (Monte Carlo)
# =============================================================================


class MonteCarloAgent:
    """モンテカルロ法によるエージェント"""

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        alpha: float = 0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.episode_memory: list[tuple[tuple[int, int], int, float | None]] = []
        # Q(s, a): 状態・行動ペアの価値
        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0.0)
        self.pi: dict[tuple[int, int], dict[int, float]] = {}

    def reset(self):
        """エピソードメモリをクリア"""
        self.episode_memory = []

    def get_action(self, state: tuple[int, int]) -> int:
        """ランダムに行動を選択"""
        return np.random.choice(self.env.get_actions())

    def add(self, state: tuple[int, int], action: int, reward: float | None):
        """経験をエピソードメモリに追加"""
        self.episode_memory.append((state, action, reward))

    def eval(self):
        """エピソードメモリを逆向きに辿ってQ(s,a)を更新（固定学習率alpha）"""
        G = 0.0
        for state, action, reward in reversed(self.episode_memory):
            r = reward if reward is not None else 0.0
            G = r + self.gamma * G
            # Q(s,a) ← Q(s,a) + alpha * (G - Q(s,a))
            self.Q[(state, action)] += self.alpha * (G - self.Q[(state, action)])

    def update(self):
        """Q(s,a)を更新し、epsilon-greedy方策でpiを更新"""
        # Q(s,a)を更新
        self.eval()

        # Q(s,a)からepsilon-greedy方策を計算してpiを更新
        n_actions = len(self.env.get_actions())
        for state in self.env.states():
            if state in self.env.walls or state == self.env.goal:
                continue

            # 各行動のQ値を取得
            action_values = {a: self.Q[(state, a)] for a in self.env.get_actions()}

            best_action = max(action_values, key=lambda a: action_values[a])
            # epsilon-greedy: 各アクションにepsilon/|A|、best_actionに追加で(1-epsilon)
            self.pi[state] = {
                a: self.epsilon / n_actions
                + (1 - self.epsilon if a == best_action else 0)
                for a in self.env.get_actions()
            }


# =============================================================================
# TD法 (Temporal Difference)
# =============================================================================


class TdAgent:
    """TD法（Q-learning）によるエージェント"""

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        alpha: float = 0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        # Q(s, a): 状態・行動ペアの価値
        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0.0)
        self.pi: dict[tuple[int, int], dict[int, float]] = {}

    def get_action(self, state: tuple[int, int]) -> int:
        """ランダムに行動を選択"""
        return np.random.choice(self.env.get_actions())

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float | None,
        next_state: tuple[int, int],
        done: bool,
    ):
        """Q値と方策を更新"""
        # TD更新: Q(s,a) ← Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        r = reward if reward is not None else 0.0
        if done:
            target = r
        else:
            # 次の状態での最大Q値
            next_q_max = max(self.Q[(next_state, a)] for a in self.env.get_actions())
            target = r + self.gamma * next_q_max
        # Q値を更新
        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])

        # 方策を更新
        if state not in self.env.walls and state != self.env.goal:
            n_actions = len(self.env.get_actions())
            action_values = {a: self.Q[(state, a)] for a in self.env.get_actions()}
            best_action = max(action_values, key=lambda a: action_values[a])
            # epsilon-greedy: 各アクションにepsilon/|A|、best_actionに追加で(1-epsilon)
            self.pi[state] = {
                a: self.epsilon / n_actions
                + (1 - self.epsilon if a == best_action else 0)
                for a in self.env.get_actions()
            }


# =============================================================================
# Q-learning (Off-policy TD)
# =============================================================================


class QLearningAgent:
    """Q-learningエージェント（Off-policy）

    - 行動選択: epsilon-greedy（探索用）
    - 目標方策: Qからgreedyに導出
    """

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        alpha: float = 0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        # Q(s, a): 状態・行動ペアの価値
        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0.0)

    def get_action(self, state: tuple[int, int]) -> int:
        """epsilon-greedy戦略でアクションを選択"""
        if np.random.random() < self.epsilon:
            # 探索: ランダムにアクションを選択
            return np.random.choice(self.env.get_actions())
        else:
            # 活用: Q値が最大のアクションを選択
            action_values = {a: self.Q[(state, a)] for a in self.env.get_actions()}
            return max(action_values, key=lambda a: action_values[a])

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float | None,
        next_state: tuple[int, int],
        done: bool,
    ):
        """Q値を更新"""
        # TD更新: Q(s,a) ← Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        r = reward if reward is not None else 0.0
        if done:
            target = r
        else:
            # 次の状態での最大Q値
            next_q_max = max(self.Q[(next_state, a)] for a in self.env.get_actions())
            target = r + self.gamma * next_q_max
        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])


# =============================================================================
# AgentTrainer
# =============================================================================


class AgentTrainer:
    """エージェントの学習を行う汎用トレーナー

    Hooks:
    - on_episode_start(agent, env): エピソード開始時に呼ばれる
    - on_step(agent, state, action, reward, next_state, done): env.step後に呼ばれる
    - on_done(agent): エピソード終了時（done=True）に呼ばれる
    - on_episode_end(agent, env, episode): エピソード終了後に呼ばれる
    """

    def __init__(
        self,
        env: GridWorld,
        agent,
        num_episodes: int = 1000,
        on_episode_start=None,
        on_step=None,
        on_done=None,
        on_episode_end=None,
    ):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.on_episode_start = on_episode_start
        self.on_step = on_step
        self.on_done = on_done
        self.on_episode_end = on_episode_end

    def train(self):
        """学習を実行"""
        for episode in range(self.num_episodes):
            state = self.env.reset()

            # エピソード開始時のhook
            if self.on_episode_start:
                self.on_episode_start(self.agent, self.env)

            while True:
                action = self.agent.get_action(state)
                next_state, reward, done = self.env.step(action)

                # ステップ後のhook
                if self.on_step:
                    self.on_step(self.agent, state, action, reward, next_state, done)

                if done:
                    # 終了時のhook
                    if self.on_done:
                        self.on_done(self.agent)
                    break

                state = next_state

            # エピソード終了後のhook
            if self.on_episode_end:
                self.on_episode_end(self.agent, self.env, episode)


UP = GridWorld.UP
DOWN = GridWorld.DOWN
LEFT = GridWorld.LEFT
RIGHT = GridWorld.RIGHT

V = defaultdict(lambda: 0)
pi = defaultdict(lambda: {UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25})


# =============================================================================
# 方策反復法 (Policy Iteration)
# =============================================================================


def eval_onestep(pi, V, env: GridWorld, gamma: float):
    """反復方策評価を1ステップ実行し、新しいVを返す"""
    new_V = defaultdict(lambda: 0)

    for state in env.states():
        # 壁とゴールはスキップ
        if state in env.walls or state == env.goal:
            continue

        action_probs = pi[state]
        value = 0

        for action, prob in action_probs.items():
            # 状態を設定してstepをシミュレート
            env.state = state
            next_state, reward, done = env.step(action)

            # 報酬がNoneの場合は0として扱う
            r = reward if reward is not None else 0
            value += prob * (r + gamma * V[next_state])

        new_V[state] = value

    return new_V


def policy_eval(pi, V, env: GridWorld, gamma: float, threshold: float = 1e-4):
    """方策評価を収束するまで繰り返す"""
    while True:
        new_V = eval_onestep(pi, V, env, gamma)

        # 更新量の最大値を計算
        max_delta = 0
        for state in env.states():
            if state in env.walls or state == env.goal:
                continue
            max_delta = max(max_delta, abs(new_V[state] - V[state]))

        V = new_V

        if max_delta < threshold:
            break

    return V


def greedy_policy(V, env: GridWorld, gamma: float):
    """価値関数Vに基づいてgreedy方策を返す"""
    new_pi = {}

    for state in env.states():
        if state in env.walls or state == env.goal:
            continue

        action_values = {}
        for action in env.get_actions():
            env.state = state
            next_state, reward, done = env.step(action)
            r = reward if reward is not None else 0
            action_values[action] = r + gamma * V[next_state]

        # 最大価値のアクションを選択
        best_action = max(action_values, key=lambda a: action_values[a])

        # greedy方策: 最大価値のアクションに確率1
        new_pi[state] = {a: 1.0 if a == best_action else 0.0 for a in env.get_actions()}

    return new_pi


def policy_iter(
    pi, V, env: GridWorld, gamma: float, threshold: float = 1e-4, on_policy_update=None
):
    """方策反復: 方策評価と方策改善を繰り返す"""
    while True:
        # 方策評価
        V = policy_eval(pi, V, env, gamma, threshold)

        # 方策改善
        new_pi = greedy_policy(V, env, gamma)

        # コールバック呼び出し
        if on_policy_update is not None:
            on_policy_update(new_pi, V)

        # 方策が変化しなくなったら終了
        unchanged = True
        for state in env.states():
            if state in env.walls or state == env.goal:
                continue
            if pi[state] != new_pi[state]:
                unchanged = False
                break

        pi = new_pi

        if unchanged:
            break

    return pi, V


# =============================================================================
# 価値反復法 (Value Iteration)
# =============================================================================


def value_iter_onestep(V, env: GridWorld, gamma: float):
    """価値反復法を1ステップ実行し、新しいVを返す"""
    new_V = defaultdict(lambda: 0)

    for state in env.states():
        if state in env.walls or state == env.goal:
            continue

        # 各アクションの価値を計算し、最大値を選択
        action_values = []
        for action in env.get_actions():
            env.state = state
            next_state, reward, done = env.step(action)
            r = reward if reward is not None else 0
            action_values.append(r + gamma * V[next_state])

        new_V[state] = max(action_values)

    return new_V


def value_iter(
    V, env: GridWorld, gamma: float, threshold: float = 1e-4, on_update=None
):
    """価値反復法を収束するまで繰り返す"""
    while True:
        new_V = value_iter_onestep(V, env, gamma)

        # 更新量の最大値を計算
        max_delta = 0
        for state in env.states():
            if state in env.walls or state == env.goal:
                continue
            max_delta = max(max_delta, abs(new_V[state] - V[state]))

        V = new_V

        # コールバック呼び出し
        if on_update is not None:
            # 現在のVからgreedy方策を導出
            pi = greedy_policy(V, env, gamma)
            on_update(pi, V)

        if max_delta < threshold:
            break

    # 最終的なgreedy方策を返す
    pi = greedy_policy(V, env, gamma)
    return pi, V


# %%
env = GridWorld()
print(env.render())

# %%
# 方策反復法
V = defaultdict(lambda: 0)
pi, V = policy_iter(
    pi, V, env, gamma=0.9, on_policy_update=lambda pi, V: env.render_v_pi(V, pi)
)

# %%
# 価値反復法
V = defaultdict(lambda: 0)
pi, V = value_iter(V, env, gamma=0.9, on_update=lambda pi, V: env.render_v_pi(V, pi))

# %%
# モンテカルロ法
env = GridWorld()
agent = MonteCarloAgent(env, gamma=0.9)

trainer = AgentTrainer(
    env,
    agent,
    num_episodes=1000,
    on_episode_start=lambda agent, env: agent.reset(),
    on_step=lambda agent, state, action, reward, next_state, done: agent.add(
        state, action, reward
    ),
    on_done=lambda agent: agent.update(),
    on_episode_end=lambda agent, env, episode: (
        env.render_q(agent.Q) if (episode + 1) % 100 == 0 else None
    ),
)
trainer.train()

# Qから最適方策とV(s)を導出して表示
V_from_Q = {s: max(agent.Q[(s, a)] for a in env.get_actions()) for s in env.states()}
pi_from_Q = {}
for state in env.states():
    if state in env.walls or state == env.goal:
        continue
    best_action = max(env.get_actions(), key=lambda a: agent.Q[(state, a)])
    pi_from_Q[state] = {a: 1.0 if a == best_action else 0.0 for a in env.get_actions()}
env.render_v_pi(V_from_Q, pi_from_Q)

# %%
# TD法 (Q-learning)
env = GridWorld()
td_agent = TdAgent(env, gamma=0.9)

trainer = AgentTrainer(
    env,
    td_agent,
    num_episodes=1000,
    on_step=lambda agent, state, action, reward, next_state, done: agent.update(
        state, action, reward, next_state, done
    ),
    on_episode_end=lambda agent, env, episode: (
        env.render_q(agent.Q) if (episode + 1) % 100 == 0 else None
    ),
)
trainer.train()

# Qから最適方策とV(s)を導出して表示
V_from_Q = {s: max(td_agent.Q[(s, a)] for a in env.get_actions()) for s in env.states()}
env.render_v_pi(V_from_Q, td_agent.pi)

# %%
# Q-learning (Off-policy)
env = GridWorld()
q_agent = QLearningAgent(env, gamma=0.9)

trainer = AgentTrainer(
    env,
    q_agent,
    num_episodes=10000,
    on_step=lambda agent, state, action, reward, next_state, done: agent.update(
        state, action, reward, next_state, done
    ),
    on_episode_end=lambda agent, env, episode: (
        env.render_q(agent.Q) if (episode + 1) % 1000 == 0 else None
    ),
)
trainer.train()

# 最終結果を表示（QからV, piを導出）
V_from_Q = {s: max(q_agent.Q[(s, a)] for a in env.get_actions()) for s in env.states()}
pi_from_Q = {}
for state in env.states():
    if state in env.walls or state == env.goal:
        continue
    best_action = max(env.get_actions(), key=lambda a: q_agent.Q[(state, a)])
    pi_from_Q[state] = {a: 1.0 if a == best_action else 0.0 for a in env.get_actions()}
env.render_v_pi(V_from_Q, pi_from_Q)

# # =============================================================================
# # RandomGenGridWorld
# # =============================================================================

# # %%
# random_env = RandomGenGridWorld(seed=42)
# print(random_env.render())

# # %%
# # 方策反復法 (RandomGenGridWorld)
# V = defaultdict(lambda: 0)
# pi = defaultdict(lambda: {UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25})
# policy_iter_count = [0]


# def on_policy_update(pi, V):
#     policy_iter_count[0] += 1
#     if policy_iter_count[0] % 3 == 0:
#         random_env.render_v_pi(V, pi)


# pi, V = policy_iter(pi, V, random_env, gamma=0.9, on_policy_update=on_policy_update)
# random_env.render_v_pi(V, pi)  # 最終結果

# # %%
# # 価値反復法 (RandomGenGridWorld)
# V = defaultdict(lambda: 0)
# value_iter_count = [0]


# def on_value_update(pi, V):
#     value_iter_count[0] += 1
#     if value_iter_count[0] % 3 == 0:
#         random_env.render_v_pi(V, pi)


# pi, V = value_iter(V, random_env, gamma=0.9, on_update=on_value_update)
# random_env.render_v_pi(V, pi)  # 最終結果
