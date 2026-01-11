# %% [markdown]
# # GridWorld DQN
# DQNを用いたGridWorldの学習

# %% Imports
import numpy as np
from collections import deque
from dataclasses import dataclass

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent_trainer import AgentTrainer, Env
from dpl.agent import ReplayBuffer, BaseAgent


# %% Feature Toggle
# 観測モード: "onehot", "local", "local_partial"
# - onehot: one-hotエンコーディング (width * height次元)
# - local: 局所視野 (3x3の壁情報9次元 + ゴールへの正規化方向ベクトル2次元 = 11次元)
# - local_partial: 部分観測 (3x3の壁情報9次元のみ、ゴール方向なし)
OBSERVATION_MODE = "local_partial"

# フレームスタック数（過去K個の観測を連結して状態とする）
# DRQNの場合は1に設定（LSTMが履歴を処理するため）
FRAME_STACK = 1

# ネットワークタイプ: "double_dqn", "dueling_dqn", "drqn"
# - double_dqn: Double DQN（action選択はqnet、Q値評価はtarget_qnet）
# - dueling_dqn: Dueling DQN（Value streamとAdvantage streamに分離）
# - drqn: Deep Recurrent Q-Network（LSTMで時系列を処理）
NETWORK_TYPE = "drqn"


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


# %% Map Rotation
def rotate_map(ascii_map: str, times: int = 1) -> str:
    """マップを時計回りに90度×times回転させる

    Args:
        ascii_map: ASCII形式のマップ文字列
        times: 回転回数（1=90度, 2=180度, 3=270度）

    Returns:
        回転後のASCII形式のマップ文字列
    """
    lines = [line.strip() for line in ascii_map.strip().split("\n")]
    grid = [line.split() for line in lines]

    for _ in range(times % 4):
        # 時計回りに90度回転: 転置してから各行を反転
        height = len(grid)
        width = len(grid[0])
        rotated = [
            [grid[height - 1 - j][i] for j in range(height)] for i in range(width)
        ]
        grid = rotated

    # ASCII文字列に変換
    return "\n".join(" ".join(row) for row in grid)


# 回転マップのテスト
print("=== Original Map ===")
print(MAP_ASCII.strip())
print("\n=== Rotated 90° ===")
print(rotate_map(MAP_ASCII, 1))
print("\n=== Rotated 180° ===")
print(rotate_map(MAP_ASCII, 2))
print("\n=== Rotated 270° ===")
print(rotate_map(MAP_ASCII, 3))
print()


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
        elif OBSERVATION_MODE == "local":
            # 3x3の壁情報(9) + 方向ベクトル(2) = 11
            return 11
        else:  # local_partial
            # 3x3の壁情報のみ = 9
            return 9

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
        elif OBSERVATION_MODE == "local":
            return self._get_state_local()
        else:  # local_partial
            return self._get_state_local_partial()

    def _get_state_onehot(self) -> np.ndarray:
        """one-hotエンコーディング"""
        state = np.zeros(self.width * self.height, dtype=np.float32)
        idx = self.agent_pos[1] * self.width + self.agent_pos[0]
        state[idx] = 1.0
        return state

    def _get_local_view(self) -> list[float]:
        """3x3の壁情報を取得（共通処理）"""
        x, y = self.agent_pos
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
        return local_view

    def _get_state_local(self) -> np.ndarray:
        """局所視野: 3x3の壁情報 + ゴールへの正規化方向ベクトル"""
        x, y = self.agent_pos
        local_view = self._get_local_view()

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

    def _get_state_local_partial(self) -> np.ndarray:
        """部分観測: 3x3の壁情報のみ（ゴール方向なし）"""
        local_view = self._get_local_view()
        return np.array(local_view, dtype=np.float32)

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
        # local_partialモードではループ検知を無効化（DRQNがLSTMで履歴を持つため）
        if OBSERVATION_MODE == "local_partial":
            truncated = self.steps >= self.max_steps
        else:
            truncated = (
                self.steps >= self.max_steps
                or self.visit_counts[pos] >= self.max_visit_count
            )

        # truncated時のペナルティ（ゴール未到達）
        if truncated and not terminated:
            reward -= 1.0

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


# %% RotatedMapGridWorld
class RotatedMapGridWorld(GridWorld):
    """リセット時にマップを90度/180度/270度回転させるGridWorld

    評価用として、回転したマップへの汎化性能をテストするために使用。
    """

    def __init__(
        self,
        base_map: str = MAP_ASCII,
        max_steps: int = 200,
        max_visit_count: int = 30,
    ):
        self._base_map = base_map
        # 回転バリエーション: 90度, 180度, 270度
        self._rotations = [1, 2, 3]
        self._current_rotation = 0

        # 初期マップ（90度回転）
        initial_map = rotate_map(base_map, self._rotations[0])
        super().__init__(
            ascii_map=initial_map,
            max_steps=max_steps,
            random_start=False,
            corner_start=True,  # 四隅からスタート
            max_visit_count=max_visit_count,
        )

    def reset(self) -> np.ndarray:
        """リセット時にランダムな回転を適用"""
        # ランダムに回転を選択
        rotation = self._rotations[np.random.randint(len(self._rotations))]
        rotated_map = rotate_map(self._base_map, rotation)
        start, goal, obstacles, width, height = parse_grid_map(rotated_map)

        # 環境パラメータを更新
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.goal = goal
        self.start = start

        # 親クラスのリセットを呼び出す
        return super().reset()


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


# %% Episode Replay Buffer (for DRQN)
@dataclass
class Transition:
    """1ステップの遷移データ"""

    o: np.ndarray  # (obs_dim,)
    a: int
    r: float
    o2: np.ndarray  # (obs_dim,)
    terminated: int  # 0/1 (bootstrap cut)


class EpisodeReplay:
    """エピソード単位でシーケンスをサンプリングするReplay Buffer

    DRQN用: LSTMの学習には連続したシーケンスが必要
    """

    def __init__(self, capacity_episodes: int = 2000):
        self.capacity = capacity_episodes
        self.episodes: list[list[Transition]] = []
        self.current: list[Transition] = []

    def __len__(self) -> int:
        """完了したエピソード数を返す"""
        return len(self.episodes)

    def start_episode(self):
        """新しいエピソードを開始"""
        self.current = []

    def add(self, o: np.ndarray, a: int, r: float, o2: np.ndarray, terminated: int):
        """遷移を現在のエピソードに追加"""
        self.current.append(Transition(o, a, r, o2, terminated))

    def end_episode(self):
        """現在のエピソードを完了してバッファに追加"""
        if len(self.current) > 0:
            self.episodes.append(self.current)
            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)
        self.current = []

    def can_sample(self, batch: int, seq_len: int, burn_in: int) -> bool:
        """サンプリング可能かどうかを判定"""
        need = seq_len + burn_in
        long_eps = [ep for ep in self.episodes if len(ep) >= need]
        return len(long_eps) >= batch

    def sample_sequences(
        self, batch: int, seq_len: int, burn_in: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """シーケンスをバッチでサンプリング

        Args:
            batch: バッチサイズ
            seq_len: 学習用シーケンス長
            burn_in: LSTM状態を安定させるためのウォームアップ長

        Returns:
            obs: (B, L, obs_dim) 観測シーケンス
            acts: (B, L) アクションシーケンス
            rews: (B, L) 報酬シーケンス
            terms: (B, L) 終了フラグシーケンス
            next_obs: (B, L, obs_dim) 次観測シーケンス
            mask: (B, L) 損失計算用マスク（burn-in部分は0）
        """
        L = burn_in + seq_len
        # 長さ十分なepisodeから選ぶ
        candidates = [ep for ep in self.episodes if len(ep) >= L]
        assert (
            len(candidates) >= batch
        ), f"Not enough episodes: {len(candidates)} < {batch}"

        obs_dim = candidates[0][0].o.shape[0]

        obs = np.zeros((batch, L, obs_dim), np.float32)
        next_obs = np.zeros((batch, L, obs_dim), np.float32)
        acts = np.zeros((batch, L), np.int64)
        rews = np.zeros((batch, L), np.float32)
        terms = np.zeros((batch, L), np.float32)
        mask = np.zeros((batch, L), np.float32)

        for b in range(batch):
            ep = candidates[np.random.randint(len(candidates))]
            start = np.random.randint(0, len(ep) - L + 1)
            chunk = ep[start : start + L]
            for t, tr in enumerate(chunk):
                obs[b, t] = tr.o
                next_obs[b, t] = tr.o2
                acts[b, t] = tr.a
                rews[b, t] = tr.r
                terms[b, t] = tr.terminated

            # burn-in を除外して loss を計算する
            mask[b, burn_in:] = 1.0

        return obs, acts, rews, terms, next_obs, mask


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


class DuelingQNet(L.Layer):
    """Dueling DQN用のQ-Network

    Q値をValue関数とAdvantage関数に分解:
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))

    これにより、アクションに依存しない状態の価値を効率的に学習できる。
    """

    def __init__(
        self, state_size: int = 2, action_size: int = 4, hidden_size: int = 64
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # 共通の特徴抽出層
        self.fc1 = L.Linear(hidden_size)
        self.fc2 = L.Linear(hidden_size)

        # Value stream: 状態価値 V(s) を出力
        self.value_fc = L.Linear(hidden_size // 2)
        self.value_out = L.Linear(1)

        # Advantage stream: 各アクションのアドバンテージ A(s, a) を出力
        self.adv_fc = L.Linear(hidden_size // 2)
        self.adv_out = L.Linear(action_size)

    def _compute_streams(self, x: Variable) -> tuple[Variable, Variable]:
        """共通層を通してValue streamとAdvantage streamを計算"""
        # 共通層
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        # Value stream
        v = F.relu(self.value_fc(h))
        v = self.value_out(v)  # (batch, 1)

        # Advantage stream
        a = F.relu(self.adv_fc(h))
        a = self.adv_out(a)  # (batch, action_size)

        return v, a

    def forward(self, *inputs: Variable) -> Variable:
        x = inputs[0]
        v, a = self._compute_streams(x)

        # Q = V + (A - mean(A))
        # mean(A)を引くことで、Vが実際の状態価値を学習しやすくなる
        a_mean = F.sum(a, axis=1, keepdims=True) / self.action_size  # (batch, 1)
        q = v + (a - a_mean)

        return q

    def get_value_and_advantage(self, x: Variable):
        """Value V(s) と Advantage A(s,a) を個別に取得（可視化用）

        Returns:
            (value, advantage): それぞれ numpy array
        """
        v, a = self._compute_streams(x)
        return v.data_required, a.data_required


class DRQN(L.Layer):
    """Deep Recurrent Q-Network

    TimeLSTMを使用して時系列データを処理するDQN。
    部分観測環境（POMDP）に対応できる。

    - forward(): バッチシーケンス処理（学習用）
    - step(): 1ステップ推論（アクション選択用）
    """

    def __init__(self, obs_dim: int = 9, action_size: int = 4, hidden_size: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_size = action_size
        self.hidden_size = hidden_size

        # 特徴抽出層
        self.fc_in = L.Linear(hidden_size, in_size=obs_dim)

        # TimeLSTM層（stateful=Trueで状態を保持）
        self.lstm = L.TimeLSTM(hidden_size, in_size=hidden_size, stateful=True)

        # Q値出力層
        self.fc_out = L.Linear(action_size, in_size=hidden_size)

    def reset_state(self):
        """LSTMの内部状態をリセット"""
        self.lstm.reset_state()

    def forward(self, *inputs: Variable) -> Variable:
        """バッチシーケンス処理（学習用）

        Args:
            inputs[0]: (B, T, obs_dim) の観測シーケンス

        Returns:
            Q_seq: (B, T, action_size) のQ値シーケンス

        Note:
            statefulモードはforward_stateful()またはstep()を使用
        """
        x = inputs[0]
        # 学習時はstateful=Falseで状態をリセット
        self.lstm.stateful = False
        self.lstm.reset_state()

        B, T, D = x.shape

        # (B*T, D) にreshape
        x2 = F.reshape(x, (B * T, D))

        # 特徴抽出
        h2 = F.relu(self.fc_in(x2))

        # (B, T, hidden) にreshape
        h = F.reshape(h2, (B, T, self.hidden_size))

        # TimeLSTM: (B, T, hidden) → (B, T, hidden)
        hs = self.lstm(h)

        # (B*T, hidden) にreshape
        hs2 = F.reshape(hs, (B * T, self.hidden_size))

        # Q値出力
        q2 = self.fc_out(hs2)

        # (B, T, action_size) にreshape
        q = F.reshape(q2, (B, T, self.action_size))

        return q

    def _forward_impl(self, x: Variable, stateful: bool) -> Variable:
        """forward実装の共通部分"""
        self.lstm.stateful = stateful
        if not stateful:
            self.lstm.reset_state()

        B, T, D = x.shape

        # (B*T, D) にreshape
        x2 = F.reshape(x, (B * T, D))

        # 特徴抽出
        h2 = F.relu(self.fc_in(x2))

        # (B, T, hidden) にreshape
        h = F.reshape(h2, (B, T, self.hidden_size))

        # TimeLSTM: (B, T, hidden) → (B, T, hidden)
        hs = self.lstm(h)

        # (B*T, hidden) にreshape
        hs2 = F.reshape(hs, (B * T, self.hidden_size))

        # Q値出力
        q2 = self.fc_out(hs2)

        # (B, T, action_size) にreshape
        q = F.reshape(q2, (B, T, self.action_size))

        return q

    def step(self, o_t: np.ndarray) -> np.ndarray:
        """1ステップ推論（アクション選択用）

        Args:
            o_t: (obs_dim,) の観測

        Returns:
            q_t: (action_size,) のQ値
        """
        # (1, 1, obs_dim) に変換
        x = Variable(o_t.astype(np.float32)[None, None, :])
        q = self._forward_impl(x, stateful=True)
        return np.array(q.data_required[0, 0])  # (action_size,)


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
        epsilon_min: float = 0.10,
        epsilon_decay: float = 0.998,  # per episode
        lr: float = 1e-3,
        batch_size: int = 32,
        buffer_size: int = 10000,
        tau: float = 0.005,  # soft update coefficient
        target_update_freq: int = 100,
        hidden_size: int = 64,
        grad_clip: float = 1.0,
        warmup_steps: int = 500,
        # DRQN用パラメータ
        seq_len: int = 20,  # 学習用シーケンス長
        burn_in: int = 20,  # LSTM安定化用のウォームアップ長
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
        # DRQN用
        self.seq_len = seq_len
        self.burn_in = burn_in

        # Q-Network（NETWORK_TYPEに応じて切り替え）
        if NETWORK_TYPE == "dueling_dqn":
            self.qnet = DuelingQNet(state_size, action_size, hidden_size)
            self.target_qnet = DuelingQNet(state_size, action_size, hidden_size)
            # ダミー入力で重みを初期化
            dummy_input = Variable(np.zeros((1, state_size), dtype=np.float32))
            self.qnet(dummy_input)
            self.target_qnet(dummy_input)
        elif NETWORK_TYPE == "drqn":
            self.qnet = DRQN(state_size, action_size, hidden_size)
            self.target_qnet = DRQN(state_size, action_size, hidden_size)
            # ダミー入力で重みを初期化（B, T, obs_dim）
            dummy_input = Variable(np.zeros((1, 1, state_size), dtype=np.float32))
            self.qnet(dummy_input)
            self.target_qnet(dummy_input)
            # LSTM状態をリセット
            self.qnet.reset_state()
            self.target_qnet.reset_state()
        else:  # double_dqn
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

        # Replay Buffer（DRQNはEpisodeReplayを使用）
        if NETWORK_TYPE == "drqn":
            self.buffer = EpisodeReplay(capacity_episodes=buffer_size // 50)
        else:
            self.buffer = ReplayBuffer(buffer_size)

        # 学習ステップカウンタ
        self.learn_step = 0

    def _soft_update_target(self):
        """Target networkをsoft updateで更新: θ' ← τθ + (1-τ)θ'"""
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

    def reset_state(self):
        """DRQN用: LSTM状態をリセット（エピソード開始時に呼び出す）"""
        if isinstance(self.qnet, DRQN):
            self.qnet.reset_state()
        if isinstance(self.target_qnet, DRQN):
            self.target_qnet.reset_state()

    def get_action(self, state: np.ndarray) -> int:
        """epsilon-greedy戦略でアクションを選択"""
        if isinstance(self.qnet, DRQN):
            # DRQN: 常にstep()でLSTM状態を更新（ランダム行動でも）
            q_values = self.qnet.step(state)
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            return int(np.argmax(q_values))
        else:
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            return self._greedy_action(state)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """指定したepsilonでアクションを選択"""
        if isinstance(self.qnet, DRQN):
            # DRQN: 常にstep()でLSTM状態を更新（ランダム行動でも）
            q_values = self.qnet.step(state)
            if np.random.random() < epsilon:
                return np.random.randint(self.action_size)
            return int(np.argmax(q_values))
        else:
            if np.random.random() < epsilon:
                return np.random.randint(self.action_size)
            return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray) -> int:
        """Q値が最大のアクションを選択"""
        if isinstance(self.qnet, DRQN):
            # DRQN: step()で1ステップ推論（状態保持）
            q_values = self.qnet.step(state)
            return int(np.argmax(q_values))
        else:
            state_var = Variable(state.reshape(1, -1).astype(np.float32))
            q_values = self.qnet(state_var)
            return int(np.argmax(q_values.data_required[0]))

    def store(self, state, action, reward, next_state, done, *, terminated=None):
        """経験をバッファに保存

        Args:
            terminated: ブートストラップカット用（DRQN用、省略時はdoneを使用）
                - terminated=True: 自然終了（ゴール到達など）→ Q(s')=0
                - terminated=False: 時間切れ（truncated）→ Q(s')を推定
        """
        if isinstance(self.buffer, EpisodeReplay):
            # DRQN: terminatedのみでブートストラップカット
            term_flag = terminated if terminated is not None else done
            self.buffer.add(state, action, reward, next_state, int(term_flag))
        else:
            self.buffer.push(state, action, reward, next_state, done)

    def start_episode(self):
        """エピソード開始時に呼び出す（DRQN用）"""
        if isinstance(self.buffer, EpisodeReplay):
            self.buffer.start_episode()

    def end_episode(self):
        """エピソード終了時に呼び出す（DRQN用）"""
        if isinstance(self.buffer, EpisodeReplay):
            self.buffer.end_episode()

    def update(self) -> float | None:
        """ミニバッチでQ-Networkを更新

        Returns:
            loss値（バッファが足りない場合やwarmup中はNone）
        """
        if isinstance(self.buffer, EpisodeReplay):
            return self._update_drqn()
        else:
            return self._update_dqn()

    def _update_dqn(self) -> float | None:
        """通常のDQN/Dueling DQN用の更新"""
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

    def _update_drqn(self) -> float | None:
        """DRQN用のシーケンスベース更新（TimeLSTM版）"""
        assert isinstance(self.buffer, EpisodeReplay)
        assert isinstance(self.qnet, DRQN)
        assert isinstance(self.target_qnet, DRQN)

        # サンプリング可能か確認
        if not self.buffer.can_sample(self.batch_size, self.seq_len, self.burn_in):
            return None

        # シーケンスをサンプリング
        obs, acts, rews, terms, next_obs, mask = self.buffer.sample_sequences(
            self.batch_size, self.seq_len, self.burn_in
        )
        # obs: (B, L, obs_dim), acts: (B, L), rews: (B, L), terms: (B, L), mask: (B, L)
        B, L, _ = obs.shape

        # Target networkでnext_obsのQ値を計算（勾配不要）
        # forward()はstateful=Falseで状態リセットされる
        target_q_seq = self.target_qnet(Variable(next_obs.astype(np.float32)))
        target_q_values = target_q_seq.data_required  # (B, L, action_size)

        # Double DQN: main networkでaction選択、target networkで評価
        main_q_for_action = self.qnet(Variable(next_obs.astype(np.float32)))
        best_actions = np.argmax(main_q_for_action.data_required, axis=2)  # (B, L)

        # Target Q値を取得
        max_next_q = np.zeros((B, L), dtype=np.float32)
        for b in range(B):
            for t in range(L):
                max_next_q[b, t] = target_q_values[b, t, best_actions[b, t]]

        # TD targets: r + γ * max_Q(s', a') * (1 - terminated)
        targets = rews + self.gamma * max_next_q * (1 - terms)  # (B, L)

        # 現在のQ値を計算（勾配計算用）
        q_seq = self.qnet(Variable(obs.astype(np.float32)))  # (B, L, action_size)

        # 選択したアクションのQ値を取得
        # acts: (B, L) → one-hot: (B, L, action_size)
        action_masks = np.eye(self.action_size)[acts]  # (B, L, action_size)
        current_q = F.sum(
            q_seq * Variable(action_masks.astype(np.float32)), axis=2
        )  # (B, L)

        # ターゲット
        targets_var = Variable(targets.astype(np.float32))  # (B, L)

        # MSE loss（マスク適用: burn-in部分は無視）
        diff = current_q - targets_var  # (B, L)
        mask_var = Variable(mask.astype(np.float32))  # (B, L)
        masked_sq_diff = diff * diff * mask_var  # (B, L)

        # 平均loss
        total_mask = np.sum(mask)
        if total_mask > 0:
            loss = F.sum(masked_sq_diff) / total_mask
        else:
            return None

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

        # LSTM状態をリセット（次のstep()呼び出しに備える）
        # forward()でバッチ処理後、hidden stateのバッチサイズが変わるため
        self.qnet.reset_state()
        self.target_qnet.reset_state()

        return float(loss.data_required)

    def decay_epsilon(self):
        """エピソード終了時にepsilonを減衰"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# %% Evaluation Function
def evaluate_agent(
    agent: DQNAgent,
    env: GridWorld | FrameStackEnv,
    num_episodes: int = 6,
):
    """学習済みエージェントを評価し、訪問したセルのポリシーを可視化

    Args:
        agent: 学習済みエージェント
        env: 環境（GridWorld or FrameStackEnv）
        num_episodes: 評価エピソード数
    """
    # ベース環境を取得
    base_env = env.env if isinstance(env, FrameStackEnv) else env

    for episode in range(num_episodes):
        # 環境をリセット（corner_start=Trueなら四隅からスタート）
        observation = env.reset()
        # DRQN用: LSTM状態をリセット
        agent.reset_state()

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

            # DRQN用: step()で1ステップ推論
            if isinstance(agent.qnet, DRQN):
                q_values = agent.qnet.step(observation)
                action = int(np.argmax(q_values))
            else:
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
base_eval_env = RotatedMapGridWorld()  # 評価は回転マップ（90°/180°/270°）

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
    num_episodes=3500,
    eval_interval=50,
    eval_n=200,
)

result = trainer.train()

# %% Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

# 1. Episode Rewards
ax1.plot(result.episode_rewards, alpha=0.7)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("Episode Rewards")
ax1.grid(True)

# 2. Average Loss
ax2.plot(result.episode_losses, alpha=0.7)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Average Loss")
ax2.set_title("Average Loss per Episode")
ax2.grid(True)

# 3. Eval Return
if result.eval_returns:
    episodes, returns = zip(*result.eval_returns)
    ax3.plot(episodes, returns, marker="o", markersize=4, linewidth=1.5)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Eval Return")
    ax3.set_title(f"Eval Return (ε=0, n={trainer.eval_n})")
    ax3.grid(True)

# 4. Success Rate
if hasattr(result, "eval_success_rates") and result.eval_success_rates:
    episodes, success_rates = zip(*result.eval_success_rates)
    ax4.plot(
        episodes,
        [r * 100 for r in success_rates],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="green",
    )
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title(f"Success Rate (n={trainer.eval_n})")
    ax4.set_ylim(0, 105)
    ax4.grid(True)

# 5. Average Steps to Goal (成功時のみ)
if hasattr(result, "eval_avg_steps") and result.eval_avg_steps:
    episodes, avg_steps = zip(*result.eval_avg_steps)
    # avg_steps=0 の場合はスキップ（成功なし）
    valid_data = [(e, s) for e, s in zip(episodes, avg_steps) if s > 0]
    if valid_data:
        valid_eps, valid_steps = zip(*valid_data)
        ax5.plot(
            valid_eps,
            valid_steps,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="orange",
        )
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Avg Steps to Goal")
    ax5.set_title("Avg Steps to Goal (success only)")
    ax5.grid(True)

# 6. 空きスロット（将来の拡張用、または非表示）
ax6.axis("off")

plt.tight_layout()
plt.show()


# %% Helper function for visualization
def _get_state_for_pos(x: int, y: int, env: GridWorld) -> np.ndarray:
    """指定位置の状態を取得（可視化用、フレームスタック対応）"""
    if OBSERVATION_MODE == "onehot":
        single_state = np.zeros(env.width * env.height, dtype=np.float32)
        single_state[y * env.width + x] = 1.0
    else:  # local or local_partial
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

        if OBSERVATION_MODE == "local":
            # ゴールへの方向ベクトル（正規化）
            gx, gy = env.goal
            dir_x = gx - x
            dir_y = gy - y
            dist = np.sqrt(dir_x**2 + dir_y**2)
            if dist > 0:
                dir_x /= dist
                dir_y /= dist
            single_state = np.array(local_view + [dir_x, dir_y], dtype=np.float32)
        else:  # local_partial
            single_state = np.array(local_view, dtype=np.float32)

    # フレームスタック: 同じ状態をK回繰り返す（静止状態を想定）
    if FRAME_STACK > 1:
        return np.tile(single_state, FRAME_STACK)
    return single_state


# %% Q-Value Helper
def _get_q_values_for_pos(
    agent: DQNAgent, x: int, y: int, env: GridWorld
) -> np.ndarray:
    """指定位置のQ値を取得（非DRQN用）

    注意: DRQNの場合は履歴なしでQ値を取得するため、
    正確なQ値ではない。DRQNには collect_q_values_from_rollouts を使用すること。
    """
    state = _get_state_for_pos(x, y, env)
    if isinstance(agent.qnet, DRQN):
        # DRQN: 履歴なしでQ値を取得（参考値のみ）
        agent.qnet.reset_state()
        state_var = Variable(state.reshape(1, 1, -1))
        return np.asarray(agent.qnet(state_var).data_required[0, 0])
    else:
        state_var = Variable(state.reshape(1, -1))
        return np.asarray(agent.qnet(state_var).data_required[0])


def collect_q_values_from_rollouts(
    agent: DQNAgent,
    env: GridWorld | FrameStackEnv,
    num_episodes: int = 20,
) -> dict[tuple[int, int], list[np.ndarray]]:
    """ロールアウトを実行して各セルでのQ値を収集（DRQN用）

    greedyポリシーで複数エピソードを実行し、各マスに到達した瞬間の
    LSTM hidden state h_t でのQ値を収集する。

    Args:
        agent: DQNAgent（DRQN qnetを持つ）
        env: 環境（GridWorld or FrameStackEnv）
        num_episodes: 収集するエピソード数

    Returns:
        位置 (x, y) -> Q値リスト [(action_size,), ...] の辞書
    """
    assert isinstance(agent.qnet, DRQN), "This function is for DRQN only"

    # ベース環境を取得
    base_env = env.env if isinstance(env, FrameStackEnv) else env

    # 位置 -> Q値リストの辞書
    q_values_by_pos: dict[tuple[int, int], list[np.ndarray]] = {}

    for _ in range(num_episodes):
        observation = env.reset()
        agent.reset_state()  # LSTM状態をリセット

        done = False
        while not done:
            # 現在位置
            pos: tuple[int, int] = (base_env.agent_pos[0], base_env.agent_pos[1])

            # step()でQ値を取得（LSTM状態が更新される）
            q_values = agent.qnet.step(observation)

            # Q値を記録
            if pos not in q_values_by_pos:
                q_values_by_pos[pos] = []
            q_values_by_pos[pos].append(q_values.copy())

            # greedy action
            action = int(np.argmax(q_values))

            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return q_values_by_pos


def aggregate_q_values(
    q_values_by_pos: dict[tuple[int, int], list[np.ndarray]],
    method: str = "mean",
) -> dict[tuple[int, int], np.ndarray]:
    """位置ごとのQ値リストを集約する

    Args:
        q_values_by_pos: 位置 -> Q値リストの辞書
        method: 集約方法 ("mean", "median", "last")

    Returns:
        位置 -> 集約されたQ値 (action_size,) の辞書
    """
    result: dict[tuple[int, int], np.ndarray] = {}

    for pos, q_list in q_values_by_pos.items():
        q_array = np.array(q_list)  # (N, action_size)
        if method == "mean":
            result[pos] = np.mean(q_array, axis=0)
        elif method == "median":
            result[pos] = np.median(q_array, axis=0)
        elif method == "last":
            result[pos] = q_list[-1]
        else:
            raise ValueError(f"Unknown method: {method}")

    return result


# %% Q-Value Heatmap (DRQN version)
def plot_q_heatmap_drqn(
    env: GridWorld | FrameStackEnv,
    q_values_by_pos: dict[tuple[int, int], np.ndarray],
):
    """DRQN用: ロールアウトで収集したQ値を4方向の三角形で可視化

    Args:
        env: 環境
        q_values_by_pos: 位置 -> 集約済みQ値 (action_size,) の辞書
    """
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors

    # ベース環境を取得
    base_env = env.env if isinstance(env, FrameStackEnv) else env
    width = base_env.width
    height = base_env.height
    obstacles = base_env.obstacles
    goal = base_env.goal
    start = base_env.start

    # Q値の範囲を取得（カラーマップ用）
    all_q_values = list(q_values_by_pos.values())
    if not all_q_values:
        print("No Q values collected from rollouts")
        return

    q_min = min(np.min(q) for q in all_q_values)
    q_max = max(np.max(q) for q in all_q_values)
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.get_cmap("RdYlGn")

    _, ax = plt.subplots(figsize=(12, 10))

    # 各セルに4つの三角形を描画
    for y in range(height):
        for x in range(width):
            # セルの中心
            cx, cy = x, y

            # 4つの三角形の頂点（セルの角と中心）
            top_tri = [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5), (cx, cy)]
            bottom_tri = [(cx - 0.5, cy + 0.5), (cx + 0.5, cy + 0.5), (cx, cy)]
            left_tri = [(cx - 0.5, cy - 0.5), (cx - 0.5, cy + 0.5), (cx, cy)]
            right_tri = [(cx + 0.5, cy - 0.5), (cx + 0.5, cy + 0.5), (cx, cy)]

            triangles = [top_tri, bottom_tri, left_tri, right_tri]
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
            elif (x, y) in q_values_by_pos:
                # ロールアウトで訪問したセル
                q_vals = q_values_by_pos[(x, y)]
                for i, (tri, offset) in enumerate(zip(triangles, label_offsets)):
                    q_val = q_vals[i]
                    color = cmap(norm(q_val))
                    poly = Polygon(
                        tri, facecolor=color, edgecolor="black", linewidth=0.5
                    )
                    ax.add_patch(poly)
                    ax.text(
                        cx + offset[0],
                        cy + offset[1],
                        f"{q_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )
            else:
                # 未訪問セルは白
                for tri in triangles:
                    poly = Polygon(
                        tri, facecolor="white", edgecolor="lightgray", linewidth=0.5
                    )
                    ax.add_patch(poly)
                ax.text(
                    cx,
                    cy,
                    "?",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="lightgray",
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
    ax.set_title("DRQN Q-Values (from rollouts with LSTM history)")

    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Q-value")

    plt.tight_layout()
    plt.show()


# %% Value and Policy (DRQN version)
def plot_value_and_policy_drqn(
    env: GridWorld | FrameStackEnv,
    q_values_by_pos: dict[tuple[int, int], np.ndarray],
):
    """DRQN用: ロールアウトで収集したQ値からV(s)とπ(s)を可視化

    Args:
        env: 環境
        q_values_by_pos: 位置 -> 集約済みQ値 (action_size,) の辞書
    """
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    # ベース環境を取得
    base_env = env.env if isinstance(env, FrameStackEnv) else env
    width = base_env.width
    height = base_env.height
    obstacles = base_env.obstacles
    goal = base_env.goal
    start = base_env.start

    if not q_values_by_pos:
        print("No Q values collected from rollouts")
        return

    # V(s) = max_a Q(s, a) と best action を計算
    v_values: dict[tuple[int, int], float] = {}
    best_actions: dict[tuple[int, int], int] = {}

    for pos, q_vals in q_values_by_pos.items():
        v_values[pos] = float(np.max(q_vals))
        best_actions[pos] = int(np.argmax(q_vals))

    # 行動に対応する矢印のオフセット
    arrow_directions = {
        0: (0, -0.3),  # 上
        1: (0, 0.3),  # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0),  # 右
    }

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # === 左: Value Heatmap ===
    v_list = list(v_values.values())
    v_min = min(v_list)
    v_max = max(v_list)
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap("RdYlGn")

    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                color = "gray"
                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax1.add_patch(rect)
            elif (x, y) in v_values:
                v = v_values[(x, y)]
                color = cmap(norm(v))
                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax1.add_patch(rect)
                ax1.text(
                    x,
                    y,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
            else:
                # 未訪問
                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="lightgray",
                    linewidth=0.5,
                )
                ax1.add_patch(rect)
                ax1.text(
                    x, y, "?", ha="center", va="center", fontsize=10, color="lightgray"
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
    ax1.set_title("DRQN V(s) = max_a Q(h_t, a) (from rollouts)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="V(s)")

    # === 右: Policy Arrows ===
    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                color = "gray"
            elif (x, y) in best_actions:
                color = "lightgreen"
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
            ax2.add_patch(rect)

            if (x, y) in best_actions and (x, y) != goal:
                action = best_actions[(x, y)]
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
    ax2.set_title("DRQN Policy π(s) = argmax_a Q(h_t, a) (from rollouts)")

    plt.tight_layout()
    plt.show()


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
            q_values[y, x] = _get_q_values_for_pos(agent, x, y, env)

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


# DRQN用のQ値収集（可視化で使い回す）
q_by_pos: dict[tuple[int, int], np.ndarray] | None = None
if NETWORK_TYPE == "drqn":
    print("\nCollecting Q-values from rollouts (DRQN)...")
    q_by_pos_raw = collect_q_values_from_rollouts(agent, env, num_episodes=20)
    q_by_pos = aggregate_q_values(q_by_pos_raw, method="mean")
    print(f"Collected Q-values for {len(q_by_pos)} positions")

print("\nQ-Value Heatmap:")
if NETWORK_TYPE == "drqn" and q_by_pos is not None:
    plot_q_heatmap_drqn(env, q_by_pos)
else:
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
            q_values[y, x] = _get_q_values_for_pos(agent, x, y, env)

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


def plot_dueling_analysis(agent: DQNAgent, env: GridWorld):
    """Dueling DQN専用: A-centered, Gap, V×Gapを横並びで可視化

    左: A(s,a) - mean(A) を4方向の三角形で表示
    中央: gap(s) = max A - second max A を表示
    右: V(s) × Gap の重ね合わせ（色=V, 明度=Gap）
    """
    if NETWORK_TYPE != "dueling_dqn":
        print("Dueling DQNが無効のため、可視化をスキップします")
        return

    from matplotlib.patches import Polygon, Rectangle
    import matplotlib.colors as mcolors

    width = env.width
    height = env.height
    obstacles = env.obstacles
    goal = env.goal
    start = env.start

    # DuelingQNetであることを確認
    if not isinstance(agent.qnet, DuelingQNet):
        print("QNetがDuelingQNetではないため、スキップします")
        return

    # 各セルのValue, Advantage, Gap値を計算
    value_values = np.zeros((height, width))
    adv_values = np.zeros((height, width, 4))
    gap_values = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            state = _get_state_for_pos(x, y, env)
            state_var = Variable(state.reshape(1, -1))
            v, adv = agent.qnet.get_value_and_advantage(state_var)
            value_values[y, x] = v[0, 0]
            # A(s,a) - mean(A) を計算
            adv_centered = adv[0] - np.mean(adv[0])
            adv_values[y, x] = adv_centered
            # gap = max - second_max
            sorted_adv = np.sort(adv_centered)[::-1]
            gap_values[y, x] = sorted_adv[0] - sorted_adv[1]

    # Figure作成（3つ横並び）
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # === 左: A-centered 4方向三角形 ===
    adv_min = np.min(adv_values)
    adv_max = np.max(adv_values)
    abs_max = max(abs(adv_min), abs(adv_max))
    adv_norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    adv_cmap = plt.get_cmap("RdBu_r")

    for y in range(height):
        for x in range(width):
            cx, cy = x, y
            # 4つの三角形
            top_tri = [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5), (cx, cy)]
            bottom_tri = [(cx - 0.5, cy + 0.5), (cx + 0.5, cy + 0.5), (cx, cy)]
            left_tri = [(cx - 0.5, cy - 0.5), (cx - 0.5, cy + 0.5), (cx, cy)]
            right_tri = [(cx + 0.5, cy - 0.5), (cx + 0.5, cy + 0.5), (cx, cy)]
            triangles = [top_tri, bottom_tri, left_tri, right_tri]
            label_offsets = [(0, -0.25), (0, 0.25), (-0.25, 0), (0.25, 0)]

            if (x, y) in obstacles:
                for tri in triangles:
                    poly = Polygon(
                        tri, facecolor="gray", edgecolor="black", linewidth=0.5
                    )
                    ax1.add_patch(poly)
                ax1.text(
                    cx,
                    cy,
                    "#",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                for i, (tri, offset) in enumerate(zip(triangles, label_offsets)):
                    adv_val = adv_values[y, x, i]
                    color = adv_cmap(adv_norm(adv_val))
                    poly = Polygon(
                        tri, facecolor=color, edgecolor="black", linewidth=0.5
                    )
                    ax1.add_patch(poly)
                    ax1.text(
                        cx + offset[0],
                        cy + offset[1],
                        f"{adv_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )

    # スタートとゴール（左）
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

    for i in range(height + 1):
        ax1.axhline(i - 0.5, color="black", linewidth=1)
    for i in range(width + 1):
        ax1.axvline(i - 0.5, color="black", linewidth=1)

    ax1.set_xlim(-0.5, width - 0.5)
    ax1.set_ylim(height - 0.5, -0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("A(s, a) - mean(A)")

    sm1 = plt.cm.ScalarMappable(cmap=adv_cmap, norm=adv_norm)
    sm1.set_array([])
    plt.colorbar(sm1, ax=ax1, label="Centered Advantage")

    # === 中央: Gap ===
    gap_min = np.min(gap_values)
    gap_max = np.max(gap_values)
    gap_cmap = plt.get_cmap("YlOrRd")

    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                rect = Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, facecolor="gray", edgecolor="black"
                )
                ax2.add_patch(rect)
                ax2.text(
                    x, y, "#", ha="center", va="center", fontsize=14, fontweight="bold"
                )
            else:
                gap_val = gap_values[y, x]
                normalized = (gap_val - gap_min) / (gap_max - gap_min + 1e-8)
                color = gap_cmap(normalized)
                rect = Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="black"
                )
                ax2.add_patch(rect)
                ax2.text(
                    x,
                    y,
                    f"{gap_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    # スタートとゴール（中央）
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
    ax2.set_title("Action Gap (max A - second max A)")

    sm2 = plt.cm.ScalarMappable(
        cmap=gap_cmap, norm=mcolors.Normalize(vmin=gap_min, vmax=gap_max)
    )
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax2, label="Gap (action confidence)")

    # === 右: V × Gap（色=V, 明度=Gap）===
    v_min = np.min(value_values)
    v_max = np.max(value_values)
    v_cmap = plt.get_cmap("RdYlGn")  # 赤(低V) → 緑(高V)

    # Gap を 0-1 に正規化（明度用）
    gap_normalized = (gap_values - gap_min) / (gap_max - gap_min + 1e-8)

    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                rect = Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, facecolor="gray", edgecolor="black"
                )
                ax3.add_patch(rect)
                ax3.text(
                    x, y, "#", ha="center", va="center", fontsize=14, fontweight="bold"
                )
            else:
                # V(s) で色を決定
                v_val = value_values[y, x]
                v_normalized = (v_val - v_min) / (v_max - v_min + 1e-8)
                base_color = np.array(v_cmap(v_normalized)[:3])

                # Gap で明度を調整（gap大 → 鮮やか、gap小 → 白っぽく）
                gap_norm = gap_normalized[y, x]
                # 明度調整: 白(1,1,1)との補間（gap小→白寄り、gap大→元の色）
                alpha_factor = 0.3 + 0.7 * gap_norm  # 0.3〜1.0
                adjusted_color = base_color * alpha_factor + np.array([1, 1, 1]) * (
                    1 - alpha_factor
                )

                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=adjusted_color,
                    edgecolor="black",
                )
                ax3.add_patch(rect)

                # V値とGap値を表示
                ax3.text(
                    x,
                    y - 0.15,
                    f"V:{v_val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
                ax3.text(
                    x,
                    y + 0.15,
                    f"G:{gap_values[y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

    # スタートとゴール（右）
    ax3.text(
        start[0],
        start[1],
        "S",
        ha="center",
        va="center",
        fontsize=10,
        color="red",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="red"),
    )
    ax3.text(
        goal[0],
        goal[1],
        "G",
        ha="center",
        va="center",
        fontsize=10,
        color="green",
        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="green"),
    )

    ax3.set_xlim(-0.5, width - 0.5)
    ax3.set_ylim(height - 0.5, -0.5)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("V(s) × Gap (color=V, brightness=Gap)")

    # カラーバー（V用）
    sm3 = plt.cm.ScalarMappable(
        cmap=v_cmap, norm=mcolors.Normalize(vmin=v_min, vmax=v_max)
    )
    sm3.set_array([])
    cbar3 = plt.colorbar(sm3, ax=ax3, label="V(s)")
    # 明度の凡例をタイトルに追加
    ax3.text(
        0.5,
        -0.12,
        "Brightness: Gap (bright=important decision)",
        transform=ax3.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    plt.show()


print("\nValue and Policy:")
if NETWORK_TYPE == "drqn" and q_by_pos is not None:
    # DRQN: 既に収集済みのQ値を使用
    plot_value_and_policy_drqn(env, q_by_pos)
else:
    plot_value_and_policy(agent, base_env)

if NETWORK_TYPE == "dueling_dqn":
    print("\nDueling DQN: Advantage & Gap Analysis:")
    plot_dueling_analysis(agent, base_env)


# %% Evaluation
print("\nEvaluating trained agent...")
evaluate_agent(agent, eval_env, num_episodes=6)
