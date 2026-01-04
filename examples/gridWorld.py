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
        done = self.state in self.reward_map  # 報酬マスに到達したら終了

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
                    ax.text(x, y, "#", ha="center", va="center", fontsize=14, color="black")
                elif state in self.reward_map:
                    ax.text(x, y, f"{self.reward_map[state]:+.1f}", ha="center", va="center", color="black")
                else:
                    # 価値を表示
                    ax.text(x, y + 0.25, f"{V[state]:.2f}", ha="center", va="center", fontsize=9, color="black")
                    # 矢印を表示（確率に応じた透明度）
                    if state in pi:
                        for action, prob in pi[state].items():
                            if prob > 0:
                                dx, dy = arrow_map[action]
                                scale = prob  # 確率に応じてスケール
                                ax.arrow(
                                    x - dx * scale / 2, y - 0.1 - dy * scale / 2,
                                    dx * scale, dy * scale,
                                    head_width=0.12 * scale, head_length=0.08 * scale,
                                    fc="black", ec="black", alpha=prob
                                )

        fig.colorbar(im, ax=ax)
        ax.set_title("Value Function & Policy")
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        plt.show()


UP = GridWorld.UP
DOWN = GridWorld.DOWN
LEFT = GridWorld.LEFT
RIGHT = GridWorld.RIGHT

V = defaultdict(lambda: 0)
pi = defaultdict(lambda: {UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25})


def eval_onestep(pi, V, env: GridWorld, gamma: float):
    """反復方策評価を1ステップ実行し、新しいVを返す"""
    new_V = defaultdict(lambda: 0)

    for state in env.states():
        # 壁と終端状態はスキップ
        if state in env.walls or state in env.reward_map:
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
            if state in env.walls or state in env.reward_map:
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
        if state in env.walls or state in env.reward_map:
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


def policy_iter(pi, V, env: GridWorld, gamma: float, threshold: float = 1e-4, on_policy_update=None):
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
            if state in env.walls or state in env.reward_map:
                continue
            if pi[state] != new_pi[state]:
                unchanged = False
                break

        pi = new_pi

        if unchanged:
            break

    return pi, V


# %%
env = GridWorld()
print(env.render())

# %%
V = defaultdict(lambda: 0)
pi, V = policy_iter(
    pi, V, env, gamma=0.9,
    on_policy_update=lambda pi, V: env.render_v_pi(V, pi)
)
