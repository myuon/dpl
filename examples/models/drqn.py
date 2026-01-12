"""DRQN (Deep Recurrent Q-Network) Agent implementation.

DQN + RNNによる部分観測環境対応
- TimeLSTMを使用した時系列処理
- EpisodeReplayによるシーケンスサンプリング
- burn-inによるLSTM状態の安定化
"""

from dataclasses import dataclass
import numpy as np

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent import BaseAgent


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


class DRQNNet(L.Layer):
    """Deep Recurrent Q-Network

    TimeLSTMを使用して時系列データを処理するDQN。
    部分観測環境（POMDP）に対応できる。

    - forward(): バッチシーケンス処理（学習用）
    - step(): 1ステップ推論（アクション選択用）
    """

    def __init__(self, obs_dim: int, action_size: int, hidden_size: int = 64):
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
            学習時はstateful=Falseで状態リセット
        """
        x = inputs[0]
        return self._forward_impl(x, stateful=False)

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


class DRQNAgent(BaseAgent):
    """DRQN Agent

    - EpisodeReplay: エピソード単位でシーケンスをサンプリング
    - TimeLSTM: 時系列処理による部分観測対応
    - burn-in: LSTM状態を安定させるためのウォームアップ
    - Double DQN: action選択はqnet、Q値評価はtarget_qnet
    """

    def __init__(
        self,
        qnet: DRQNNet,
        target_qnet: DRQNNet,
        action_size: int,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 32,
        buffer_size: int = 2000,  # エピソード数
        tau: float = 0.005,
        warmup_episodes: int = 20,  # エピソード数でのwarmup
        seq_len: int = 20,
        burn_in: int = 10,
        grad_clip: float = 1.0,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.warmup_episodes = warmup_episodes
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.grad_clip = grad_clip

        # Q-Network
        self.qnet = qnet
        self.target_qnet = target_qnet

        # Target networkを初期化
        self._hard_update_target()

        # Optimizer
        self.optimizer = Adam(lr=lr).setup(self.qnet)

        # Episode Replay Buffer
        self.buffer = EpisodeReplay(capacity_episodes=buffer_size)

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
        """LSTM状態をリセット（エピソード開始時に呼び出す）"""
        self.qnet.reset_state()
        self.target_qnet.reset_state()

    def get_action(self, state: np.ndarray) -> int:
        """epsilon-greedy戦略でアクションを選択

        重要: ランダム行動でもstep()を呼んでLSTM状態を更新する
        """
        # 常にstep()でLSTM状態を更新
        q_values = self.qnet.step(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(q_values))

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """アクションを選択"""
        if explore:
            return self.get_action(state)
        # greedy行動でもstep()でLSTM状態を更新
        q_values = self.qnet.step(state)
        return int(np.argmax(q_values))

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
        """経験をバッファに保存

        Args:
            terminated: ブートストラップカット用
                - terminated=True: 自然終了（ゴール到達など）→ Q(s')=0
                - terminated=False: 時間切れ（truncated）→ Q(s')を推定
        """
        term_flag = terminated if terminated is not None else done
        self.buffer.add(state, action, reward, next_state, int(term_flag))

    def start_episode(self):
        """エピソード開始時に呼び出す"""
        self.buffer.start_episode()

    def end_episode(self):
        """エピソード終了時に呼び出す"""
        self.buffer.end_episode()

    def update(self) -> dict | None:
        """シーケンスベースでQ-Networkを更新"""
        # Warmup期間中は更新しない
        if len(self.buffer) < self.warmup_episodes:
            return None

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
        self._soft_update_target()

        # LSTM状態をリセット（次のstep()呼び出しに備える）
        # forward()でバッチ処理後、hidden stateのバッチサイズが変わるため
        self.qnet.reset_state()
        self.target_qnet.reset_state()

        self.last_loss = float(loss.data_required)
        return {"loss": self.last_loss}

    def decay_epsilon(self):
        """エピソード終了時にepsilonを減衰"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def drqn_stats_extractor(agent) -> str | None:
    """DRQN Agent用の統計抽出関数"""
    epsilon = getattr(agent, "epsilon", None)
    loss = getattr(agent, "last_loss", None)
    buffer_len = len(agent.buffer) if hasattr(agent, "buffer") else None

    parts = []
    if epsilon is not None:
        parts.append(f"ε={epsilon:.3f}")
    if loss is not None:
        parts.append(f"L={loss:.4f}")
    if buffer_len is not None:
        parts.append(f"Eps={buffer_len}")

    if not parts:
        return None
    return ", " + ", ".join(parts)
