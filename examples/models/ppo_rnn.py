"""PPO + RNN (Recurrent PPO) Agent implementation.

PPO with LSTM for partial observation environments:
- EpisodeRolloutBuffer: Episode-based trajectory storage
- RecurrentActorCritic: MLP → LSTM → Actor/Critic heads
- PPORNNAgent: DRQN-style state management + PPO update logic

Key differences from vanilla PPO:
- Episode boundary handling (hidden state reset)
- Sequence-based minibatching (no random shuffling that breaks temporal structure)
- Burn-in for LSTM warm-up
"""

from dataclasses import dataclass
import numpy as np

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam
from dpl.agent import BaseAgent


def log_softmax(logits: Variable, axis: int = 1) -> Variable:
    """log(softmax(x)) を数値的に安定に計算"""
    x_max = F.max(logits, axis=axis, keepdims=True)
    x_shifted = logits - x_max
    log_sum_exp = F.log(F.sum(F.exp(x_shifted), axis=axis, keepdims=True))
    return x_shifted - log_sum_exp


@dataclass
class PPOTransition:
    """1ステップの遷移データ（PPO用）"""
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float


class EpisodeRolloutBuffer:
    """エピソード単位でtrajectoryを保存（RNN用）

    DRQNのEpisodeReplayを参考に、PPO用のデータ（log_prob, value）も保存

    - エピソード境界を保持
    - GAEはエピソード単位で計算
    - シーケンスを切らずにミニバッチ化
    - step数ベースで更新タイミングを判定
    - n_envs > 1 の場合は並列環境からのデータを管理
    """

    def __init__(self, max_episodes: int = 100, n_envs: int = 1):
        self.max_episodes = max_episodes
        self.n_envs = n_envs
        self.episodes: list[list[PPOTransition]] = []
        self._total_steps = 0  # 完了エピソードの総step数

        # 単一環境用
        self.current: list[PPOTransition] = []

        # 並列環境用: 各環境ごとの現在エピソード
        self.current_parallel: list[list[PPOTransition]] = [[] for _ in range(n_envs)]

        # 各エピソードの last_value (未完了エピソードのブートストラップ用)
        # done=True で終了したエピソードは 0.0、未完了は V(s_last)
        self.episode_last_values: list[float] = []

        # GAE計算後のデータ
        self.computed_data: list[dict] = []

    def __len__(self) -> int:
        """完了したエピソード数を返す"""
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        """完了エピソードの総step数を返す"""
        return self._total_steps

    def start_episode(self):
        """新しいエピソードを開始（単一環境用）"""
        self.current = []

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            log_prob: float, value: float):
        """遷移を現在のエピソードに追加（単一環境用）"""
        self.current.append(PPOTransition(obs, action, reward, done, log_prob, value))

    def end_episode(self, env_id: int | None = None):
        """現在のエピソードを完了してバッファに追加

        Args:
            env_id: 並列環境の場合、終了する環境ID
                    None の場合は単一環境モード
        """
        if env_id is None:
            # 単一環境モード
            if len(self.current) > 0:
                self._total_steps += len(self.current)
                self.episodes.append(self.current)
                self.episode_last_values.append(0.0)  # 完了エピソード
                if len(self.episodes) > self.max_episodes:
                    removed = self.episodes.pop(0)
                    self.episode_last_values.pop(0)
                    self._total_steps -= len(removed)
            self.current = []
        else:
            # 並列環境モード
            if len(self.current_parallel[env_id]) > 0:
                self._total_steps += len(self.current_parallel[env_id])
                self.episodes.append(self.current_parallel[env_id])
                self.episode_last_values.append(0.0)  # 完了エピソード
                if len(self.episodes) > self.max_episodes:
                    removed = self.episodes.pop(0)
                    self.episode_last_values.pop(0)
                    self._total_steps -= len(removed)
            self.current_parallel[env_id] = []

    def add_parallel(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
    ):
        """並列環境からのデータを追加

        Args:
            obs: (n_envs, obs_dim)
            actions: (n_envs,)
            rewards: (n_envs,)
            dones: (n_envs,)
            log_probs: (n_envs,)
            values: (n_envs,)
        """
        for i in range(self.n_envs):
            self.current_parallel[i].append(PPOTransition(
                obs=obs[i],
                action=int(actions[i]),
                reward=float(rewards[i]),
                done=bool(dones[i]),
                log_prob=float(log_probs[i]),
                value=float(values[i]),
            ))

    def finalize_rollout(self, last_values: np.ndarray | None = None):
        """rollout終了時に未完了エピソードも保存（途中打ち切り）

        並列環境でrollout_lenステップ収集後に呼ぶ

        Args:
            last_values: (n_envs,) 各環境の最終観測から計算した V(s)
                         未完了エピソードのブートストラップに使用
        """
        for i in range(self.n_envs):
            if len(self.current_parallel[i]) > 0:
                self._total_steps += len(self.current_parallel[i])
                self.episodes.append(self.current_parallel[i])
                # 未完了エピソードは last_value でブートストラップ
                if last_values is not None:
                    self.episode_last_values.append(float(last_values[i]))
                else:
                    self.episode_last_values.append(0.0)
                self.current_parallel[i] = []

    def compute_gae_all(self, gamma: float, gae_lambda: float):
        """全エピソードのGAEを計算

        完了エピソード: last_value=0 でブートストラップ
        未完了エピソード: last_value=V(s_last) でブートストラップ
        """
        self.computed_data = []

        for ep_idx, ep in enumerate(self.episodes):
            n = len(ep)
            if n == 0:
                continue

            obs = np.array([t.obs for t in ep], dtype=np.float32)
            actions = np.array([t.action for t in ep], dtype=np.int64)
            rewards = np.array([t.reward for t in ep], dtype=np.float32)
            dones = np.array([t.done for t in ep], dtype=np.float32)
            log_probs = np.array([t.log_prob for t in ep], dtype=np.float32)
            values = np.array([t.value for t in ep], dtype=np.float32)

            # 未完了エピソードの last_value を取得
            last_value = self.episode_last_values[ep_idx] if ep_idx < len(self.episode_last_values) else 0.0

            # GAE計算
            advantages = np.zeros(n, dtype=np.float32)
            gae = 0.0

            for t in reversed(range(n)):
                if t == n - 1:
                    next_value = last_value  # 完了なら0、未完了ならV(s_last)
                else:
                    next_value = values[t + 1]

                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                advantages[t] = gae

            returns = advantages + values

            self.computed_data.append({
                "obs": obs,
                "actions": actions,
                "log_probs": log_probs,
                "advantages": advantages,
                "returns": returns,
            })

    def can_sample(self, batch_size: int, seq_len: int, burn_in: int) -> bool:
        """サンプリング可能かどうかを判定"""
        L = seq_len + burn_in
        # 全エピソードからサンプル可能なシーケンス数をカウント
        total_sequences = 0
        for d in self.computed_data:
            ep_len = len(d["obs"])
            if ep_len >= L:
                total_sequences += ep_len - L + 1
        return total_sequences >= batch_size

    def get_sequence_batches(self, batch_size: int, seq_len: int, burn_in: int):
        """シーケンス単位でミニバッチを返す

        DRQNのsample_sequencesを参考に、PPO用のデータを返す
        各エピソードから複数のシーケンスをサンプル可能

        Yields:
            obs: (B, L, obs_dim)
            actions: (B, L)
            log_probs: (B, L)
            advantages: (B, L)
            returns: (B, L)
            mask: (B, L) - burn_in部分は0
        """
        L = burn_in + seq_len

        # 各エピソードからサンプル可能なシーケンスの開始位置を列挙
        # (data_idx, start_pos) のリスト
        all_sequences: list[tuple[int, int]] = []
        for data_idx, data in enumerate(self.computed_data):
            ep_len = len(data["obs"])
            if ep_len >= L:
                # このエピソードからサンプル可能な全開始位置
                for start in range(ep_len - L + 1):
                    all_sequences.append((data_idx, start))

        if len(all_sequences) < batch_size:
            return

        obs_dim = self.computed_data[0]["obs"].shape[1]

        # 全シーケンスをシャッフル
        np.random.shuffle(all_sequences)

        for batch_start in range(0, len(all_sequences), batch_size):
            batch_sequences = all_sequences[batch_start:batch_start + batch_size]
            if len(batch_sequences) < batch_size:
                break

            B = len(batch_sequences)
            obs = np.zeros((B, L, obs_dim), np.float32)
            actions = np.zeros((B, L), np.int64)
            log_probs = np.zeros((B, L), np.float32)
            advantages = np.zeros((B, L), np.float32)
            returns = np.zeros((B, L), np.float32)
            mask = np.zeros((B, L), np.float32)

            for b, (data_idx, start) in enumerate(batch_sequences):
                data = self.computed_data[data_idx]

                obs[b] = data["obs"][start:start + L]
                actions[b] = data["actions"][start:start + L]
                log_probs[b] = data["log_probs"][start:start + L]
                advantages[b] = data["advantages"][start:start + L]
                returns[b] = data["returns"][start:start + L]

                # burn_in部分を除外
                mask[b, burn_in:] = 1.0

            yield obs, actions, log_probs, advantages, returns, mask

    def clear(self):
        """バッファをクリア（update後に呼ぶ）"""
        self.episodes = []
        self.episode_last_values = []
        self.current = []
        self.current_parallel = [[] for _ in range(self.n_envs)]
        self.computed_data = []
        self._total_steps = 0


class ParallelRolloutBuffer:
    """(rollout_len, n_envs)形式でrolloutを保持（ベクトル化版）

    収集時は固定サイズ配列にベクトル代入（forループなし）
    学習時はエピソード単位に変換してシーケンスバッチ抽出
    """

    def __init__(self, rollout_len: int, n_envs: int, obs_dim: int):
        self.rollout_len = rollout_len
        self.n_envs = n_envs
        self.obs_dim = obs_dim

        # 固定サイズの配列
        self.obs = np.zeros((rollout_len, n_envs, obs_dim), np.float32)
        self.actions = np.zeros((rollout_len, n_envs), np.int64)
        self.rewards = np.zeros((rollout_len, n_envs), np.float32)
        self.dones = np.zeros((rollout_len, n_envs), np.float32)
        self.log_probs = np.zeros((rollout_len, n_envs), np.float32)
        self.values = np.zeros((rollout_len, n_envs), np.float32)
        self.episode_starts = np.zeros((rollout_len, n_envs), np.float32)

        # GAE計算後
        self.advantages = np.zeros((rollout_len, n_envs), np.float32)
        self.returns = np.zeros((rollout_len, n_envs), np.float32)

        self.ptr = 0

    @property
    def total_steps(self) -> int:
        """現在のバッファ内のステップ数"""
        return self.ptr * self.n_envs

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        episode_starts: np.ndarray,
    ):
        """1ステップ分を追加（ベクトル代入、forループなし）

        Args:
            obs: (n_envs, obs_dim)
            actions: (n_envs,)
            rewards: (n_envs,)
            dones: (n_envs,)
            log_probs: (n_envs,)
            values: (n_envs,)
            episode_starts: (n_envs,) - このステップがエピソード開始か
        """
        if self.ptr >= self.rollout_len:
            raise IndexError(
                f"Buffer overflow: ptr={self.ptr} >= rollout_len={self.rollout_len}. "
                f"Did you forget to call clear()?"
            )
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.episode_starts[self.ptr] = episode_starts
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """(rollout_len, n_envs)形式でGAE計算（ベクトル化）

        Args:
            last_values: (n_envs,) 最終観測から計算したV(s)
            gamma: 割引率
            gae_lambda: GAEラムダ
        """
        gae = np.zeros(self.n_envs, np.float32)

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def extract_episodes(self) -> tuple[list[dict], list[float]]:
        """(rollout_len, n_envs)からエピソード単位に変換

        Returns:
            episodes: list of dict (obs, actions, log_probs, advantages, returns)
            episode_last_values: 各エピソードのlast_value（完了なら0、未完了ならV(s)）
        """
        episodes = []
        episode_last_values = []

        for env_id in range(self.n_envs):
            # この環境のデータを抽出
            env_obs = self.obs[:self.ptr, env_id]
            env_actions = self.actions[:self.ptr, env_id]
            env_log_probs = self.log_probs[:self.ptr, env_id]
            env_advantages = self.advantages[:self.ptr, env_id]
            env_returns = self.returns[:self.ptr, env_id]
            env_dones = self.dones[:self.ptr, env_id]
            env_starts = self.episode_starts[:self.ptr, env_id]

            # エピソード境界を検出
            start_indices = np.where(env_starts == 1.0)[0]
            if len(start_indices) == 0:
                start_indices = np.array([0])
            elif start_indices[0] != 0:
                start_indices = np.concatenate([[0], start_indices])

            # 各エピソードを抽出
            for i, start in enumerate(start_indices):
                if i + 1 < len(start_indices):
                    end = start_indices[i + 1]
                else:
                    end = self.ptr

                if end <= start:
                    continue

                episodes.append({
                    "obs": env_obs[start:end].copy(),
                    "actions": env_actions[start:end].copy(),
                    "log_probs": env_log_probs[start:end].copy(),
                    "advantages": env_advantages[start:end].copy(),
                    "returns": env_returns[start:end].copy(),
                })

                # 完了エピソードか未完了エピソードか
                if i + 1 < len(start_indices):
                    # 次のエピソードがある = このエピソードは完了
                    episode_last_values.append(0.0)
                else:
                    # 最後のエピソード = rollout末端で打ち切り = last_valueでブートストラップ済み
                    # GAE計算時にlast_valuesを使っているので、ここでは0でOK
                    episode_last_values.append(0.0)

        return episodes, episode_last_values

    def can_sample(self, batch_size: int, seq_len: int, burn_in: int = 0) -> bool:
        """サンプリング可能か判定

        batch_sizeはシーケンス数と比較する（エピソード数ではない）
        """
        episodes, _ = self.extract_episodes()
        L = burn_in + seq_len
        # 全エピソードからサンプル可能なシーケンス数をカウント
        total_sequences = 0
        for ep in episodes:
            ep_len = len(ep["obs"])
            if ep_len >= L:
                total_sequences += ep_len - L + 1
        return total_sequences >= batch_size

    def get_sequence_batches(
        self,
        batch_size: int,
        seq_len: int,
        burn_in: int = 0,
    ):
        """シーケンスバッチを生成（既存ロジックを活用）

        Yields:
            obs: (B, L, obs_dim)
            actions: (B, L)
            log_probs: (B, L)
            advantages: (B, L)
            returns: (B, L)
            mask: (B, L) - burn_in部分は0
        """
        episodes, _ = self.extract_episodes()
        L = burn_in + seq_len

        # サンプル可能なシーケンス位置を列挙
        all_sequences = []
        for ep_idx, ep in enumerate(episodes):
            ep_len = len(ep["obs"])
            if ep_len >= L:
                for start in range(ep_len - L + 1):
                    all_sequences.append((ep_idx, start))

        if len(all_sequences) == 0:
            return

        np.random.shuffle(all_sequences)

        for batch_start in range(0, len(all_sequences), batch_size):
            batch_sequences = all_sequences[batch_start:batch_start + batch_size]
            B = len(batch_sequences)

            obs = np.zeros((B, L, self.obs_dim), np.float32)
            actions = np.zeros((B, L), np.int64)
            log_probs = np.zeros((B, L), np.float32)
            advantages = np.zeros((B, L), np.float32)
            returns = np.zeros((B, L), np.float32)
            mask = np.zeros((B, L), np.float32)

            for b, (ep_idx, start) in enumerate(batch_sequences):
                ep = episodes[ep_idx]
                obs[b] = ep["obs"][start:start + L]
                actions[b] = ep["actions"][start:start + L]
                log_probs[b] = ep["log_probs"][start:start + L]
                advantages[b] = ep["advantages"][start:start + L]
                returns[b] = ep["returns"][start:start + L]
                mask[b, burn_in:] = 1.0

            yield obs, actions, log_probs, advantages, returns, mask

    def clear(self):
        """バッファをクリア"""
        self.ptr = 0
        # 配列は再利用（ゼロクリア不要、ptrで管理）


class RecurrentActorCritic(L.Layer):
    """LSTM付きActor-Critic

    Architecture:
        obs → MLP → feature → LSTM → h → (policy_logits, value)

    DRQNNetを参考に、stateful/statelessの切り替えに対応
    n_envs > 1 の場合は並列環境用のLSTM状態管理を行う
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64, n_envs: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.n_envs = n_envs

        # Feature extractor (MLP)
        self.feature = L.Linear(hidden_size, in_size=obs_dim)

        # LSTM
        self.lstm = L.TimeLSTM(hidden_size, in_size=hidden_size, stateful=True)

        # Actor/Critic heads
        self.actor_head = L.Linear(action_dim, in_size=hidden_size)
        self.critic_head = L.Linear(1, in_size=hidden_size)

        # 並列環境用のLSTM状態（n_envs > 1の場合に使用）
        self._h_states: np.ndarray | None = None  # (n_envs, hidden_size)
        self._c_states: np.ndarray | None = None  # (n_envs, hidden_size)

    def reset_state(self, env_id: int | None = None):
        """LSTMの内部状態をリセット

        Args:
            env_id: 特定環境のみリセットする場合に指定
                    None の場合は全環境リセット
        """
        if self.n_envs == 1:
            # 単一環境: 従来の動作
            self.lstm.reset_state()
        else:
            # 並列環境
            if env_id is None:
                # 全環境リセット
                self._h_states = np.zeros((self.n_envs, self.hidden_size), dtype=np.float32)
                self._c_states = np.zeros((self.n_envs, self.hidden_size), dtype=np.float32)
            else:
                # 特定環境のみリセット
                if self._h_states is None:
                    self._h_states = np.zeros((self.n_envs, self.hidden_size), dtype=np.float32)
                    self._c_states = np.zeros((self.n_envs, self.hidden_size), dtype=np.float32)
                self._h_states[env_id] = 0
                self._c_states[env_id] = 0

    def forward(self, *inputs: Variable) -> tuple[Variable, Variable]:
        """バッチシーケンス処理（学習用）

        Args:
            inputs[0]: (B, T, obs_dim) の観測シーケンス

        Returns:
            logits: (B, T, action_dim)
            values: (B, T, 1)
        """
        x = inputs[0]
        return self._forward_impl(x, stateful=False)

    def _forward_impl(self, x: Variable, stateful: bool) -> tuple[Variable, Variable]:
        """forward実装の共通部分"""
        self.lstm.stateful = stateful
        if not stateful:
            self.lstm.reset_state()

        B, T, D = x.shape

        # (B*T, D) にreshape
        x2 = F.reshape(x, (B * T, D))

        # Feature extraction
        h2 = F.relu(self.feature(x2))

        # (B, T, hidden) にreshape
        h = F.reshape(h2, (B, T, self.hidden_size))

        # TimeLSTM: (B, T, hidden) → (B, T, hidden)
        hs = self.lstm(h)

        # (B*T, hidden) にreshape
        hs2 = F.reshape(hs, (B * T, self.hidden_size))

        # Actor/Critic heads
        logits2 = self.actor_head(hs2)
        values2 = self.critic_head(hs2)

        # Reshape to (B, T, dim)
        logits = F.reshape(logits2, (B, T, self.action_dim))
        values = F.reshape(values2, (B, T, 1))

        return logits, values

    def step(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """1ステップ推論（stateful=True）- 単一環境用

        Args:
            obs: (obs_dim,) の観測

        Returns:
            logits: (action_dim,) のlogits
            value: スカラー値
        """
        # (1, 1, obs_dim) に変換
        x = Variable(obs.astype(np.float32)[None, None, :])
        logits, value = self._forward_impl(x, stateful=True)
        return logits.data_required[0, 0], float(value.data_required[0, 0, 0])

    def steps(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """並列ステップ推論（n_envs環境同時）

        Args:
            obs: (n_envs, obs_dim) の観測

        Returns:
            logits: (n_envs, action_dim) のlogits
            values: (n_envs,) のvalue
        """
        assert self.n_envs > 1, "steps() is for parallel envs (n_envs > 1)"

        # 状態が未初期化なら初期化
        if self._h_states is None:
            self.reset_state()

        # LSTM状態を設定
        self.lstm.lstm.h = Variable(self._h_states.astype(np.float32))
        self.lstm.lstm.c = Variable(self._c_states.astype(np.float32))

        # (n_envs, obs_dim) → (n_envs, 1, obs_dim)
        x = Variable(obs[:, None, :].astype(np.float32))
        logits, values = self._forward_impl(x, stateful=True)

        # 状態を保存
        self._h_states = self.lstm.lstm.h.data.astype(np.float32)
        self._c_states = self.lstm.lstm.c.data.astype(np.float32)

        return logits.data_required[:, 0, :], values.data_required[:, 0, 0]

    def evaluate_sequences(
        self, obs: Variable, actions: np.ndarray
    ) -> tuple[Variable, Variable, Variable]:
        """シーケンスバッチ評価

        Args:
            obs: (B, T, obs_dim) の観測シーケンス
            actions: (B, T) のアクション

        Returns:
            log_probs: (B, T)
            entropy: (B, T)
            values: (B, T)
        """
        logits, values = self(obs)  # stateful=False

        B, T, A = logits.shape

        # Reshape for softmax: (B*T, A)
        logits_2d = F.reshape(logits, (B * T, A))
        probs_2d = F.softmax(logits_2d)
        log_probs_2d = log_softmax(logits_2d)

        # 選択されたアクションのlog_prob
        actions_flat = actions.reshape(-1)
        action_masks = np.eye(A)[actions_flat.astype(int)]
        action_log_probs_2d = F.sum(
            log_probs_2d * Variable(action_masks.astype(np.float32)), axis=1
        )

        # Entropy: -Σ p log p
        entropy_2d = -F.sum(probs_2d * log_probs_2d, axis=1)

        # Reshape back to (B, T)
        log_probs = F.reshape(action_log_probs_2d, (B, T))
        entropy = F.reshape(entropy_2d, (B, T))
        values_flat = F.reshape(values, (B, T))

        return log_probs, entropy, values_flat


class PPORNNAgent(BaseAgent):
    """PPO + RNN Agent

    DRQNのRNN管理パターン + PPOの更新ロジック

    Key features:
    - Episode-based buffer (EpisodeRolloutBuffer)
    - LSTM state management (reset_state, start/end_episode)
    - Sequence-based minibatching with burn-in
    - PPO clipped objective
    - n_envs > 1 で並列環境対応
    """

    def __init__(
        self,
        actor_critic: RecurrentActorCritic,
        action_size: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.02,  # RNN用に高め
        value_coef: float = 0.5,
        lr: float = 3e-4,
        n_epochs: int = 4,
        seq_len: int = 16,
        burn_in: int = 4,
        batch_size: int = 16,
        rollout_steps: int = 2048,  # step数ベースで更新（episode数ではなく）
        max_episodes: int = 100,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,  # KL超過で早期停止
        n_envs: int = 1,  # 並列環境数
        rollout_len: int = 256,  # 並列環境でのrollout長
    ):
        self.actor_critic = actor_critic
        self.action_size = action_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.n_envs = n_envs
        self.rollout_len = rollout_len

        self.optimizer = Adam(lr=lr).setup(self.actor_critic)
        # n_envs > 1 の場合はベクトル化されたバッファを使用
        self.buffer = ParallelRolloutBuffer(
            rollout_len=rollout_len, n_envs=n_envs, obs_dim=obs_dim
        )

        # 内部状態（単一環境用）
        self._last_log_prob = 0.0
        self._last_value = 0.0

        # 並列環境用
        self._last_log_probs: np.ndarray = np.zeros(n_envs)
        self._last_values: np.ndarray = np.zeros(n_envs)

        # モニタリング用
        self.last_policy_loss: float | None = None
        self.last_value_loss: float | None = None
        self.last_entropy: float | None = None
        self.last_approx_kl: float | None = None
        self.last_clip_fraction: float | None = None
        self.last_explained_var: float | None = None
        self.last_total_steps: int | None = None  # 更新時のstep数

    def reset_state(self, env_id: int | None = None):
        """LSTM状態をリセット

        Args:
            env_id: 並列環境の場合、特定環境のみリセット
                    None の場合は全環境リセット
        """
        self.actor_critic.reset_state(env_id)

    def start_episode(self):
        """エピソード開始時に呼び出す"""
        self.buffer.start_episode()

    def end_episode(self):
        """エピソード終了時に呼び出す"""
        self.buffer.end_episode()

    def get_action(self, state: np.ndarray) -> int:
        """Stateful推論でアクションを選択（LSTM状態を維持）"""
        logits, value = self.actor_critic.step(state)

        # Softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        action = np.random.choice(self.action_size, p=probs)
        log_prob = float(np.log(probs[action] + 1e-8))

        self._last_log_prob = log_prob
        self._last_value = value
        return action

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """アクションを選択

        PPO+RNNは常に確率的ポリシーを使用
        explore=Falseでも確率的サンプリング（ただしLSTM状態は更新）
        """
        return self.get_action(state)

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
        """経験をバッファに保存（単一環境用）"""
        self.buffer.add(
            state, action, reward, done,
            self._last_log_prob, self._last_value
        )

    # ============ 並列環境用メソッド ============

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """並列行動選択

        Args:
            states: (n_envs, obs_dim)

        Returns:
            actions: (n_envs,)
        """
        logits, values = self.actor_critic.steps(states)  # (n_envs, action_dim), (n_envs,)

        # 各環境で独立にサンプリング
        # Softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (n_envs, action_dim)

        actions = np.array([
            np.random.choice(self.action_size, p=p) for p in probs
        ])

        # log_probを保存
        self._last_log_probs = np.log(probs[np.arange(self.n_envs), actions] + 1e-8)
        self._last_values = values

        return actions

    def store_parallel(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        terminateds: np.ndarray,
    ):
        """並列データ保存

        Args:
            states: (n_envs, obs_dim)
            actions: (n_envs,)
            rewards: (n_envs,)
            next_states: (n_envs, obs_dim) - 未使用だが互換性のため
            dones: (n_envs,)
            terminateds: (n_envs,) - エピソード終了判定用
        """
        self.buffer.add_parallel(
            obs=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=self._last_log_probs,
            values=self._last_values,
        )

        # エピソード終了処理
        for i in range(self.n_envs):
            if dones[i]:
                self.buffer.end_episode(env_id=i)

    def collect_rollout(self, env) -> tuple[int, list[float]]:
        """n_envs × rollout_len ステップのデータを収集（ベクトル化版）

        Args:
            env: VectorEnvWrapper

        Returns:
            total_steps: 収集したステップ数
            episode_rewards: 完了したエピソードのリワードリスト
        """
        # バッファをクリア（新しいrollout開始）
        self.buffer.clear()

        obs = env.reset()  # (n_envs, obs_dim)
        self.reset_state()  # 全環境のLSTM状態リセット

        episode_rewards: list[float] = []
        current_rewards = np.zeros(self.n_envs)
        episode_starts = np.ones(self.n_envs, dtype=np.float32)  # 最初は全環境がエピソード開始

        for step in range(self.rollout_len):
            actions = self.get_actions(obs)  # (n_envs,)
            next_obs, rewards, terminateds, truncateds = env.step(actions)
            dones = terminateds | truncateds

            # ベクトル化された保存（forループなし）
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
                # 完了エピソードのリワードを記録
                episode_rewards.extend(current_rewards[done_mask].tolist())
                current_rewards[done_mask] = 0.0

                # LSTM状態リセット（環境ごとに必要）
                for i in np.where(done_mask)[0]:
                    next_obs[i] = env.reset_single(i)
                    self.reset_state(env_id=i)

            obs = next_obs
            # 次のステップでエピソード開始になる環境をマーク
            episode_starts = dones.astype(np.float32)

        # 未完了エピソードの最終観測から V(s) を計算（ブートストラップ用）
        _, last_values = self.actor_critic.steps(obs)

        # GAE計算（ベクトル化）
        self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

        return self.n_envs * self.rollout_len, episode_rewards

    def update(self) -> dict | None:
        """step数が十分たまったら更新（episode数ではなくstep数ベース）"""
        # step数ベースで更新判定
        if self.buffer.total_steps < self.rollout_steps:
            return None

        # GAE計算はcollect_rollout()で実行済み

        # サンプリング可能か確認
        if not self.buffer.can_sample(self.batch_size, self.seq_len, self.burn_in):
            return None

        self.last_total_steps = self.buffer.total_steps

        # n_epochs × シーケンスバッチ更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_explained_var = 0.0
        n_updates = 0
        early_stopped = False

        for epoch in range(self.n_epochs):
            if early_stopped:
                break
            for batch in self.buffer.get_sequence_batches(
                self.batch_size, self.seq_len, self.burn_in
            ):
                loss_dict = self._update_sequence_batch(batch)
                total_policy_loss += loss_dict["policy_loss"]
                total_value_loss += loss_dict["value_loss"]
                total_entropy += loss_dict["entropy"]
                total_approx_kl += loss_dict["approx_kl"]
                total_clip_fraction += loss_dict["clip_fraction"]
                total_explained_var += loss_dict["explained_var"]
                n_updates += 1

                # target_kl による早期停止
                if self.target_kl is not None and loss_dict["approx_kl"] > self.target_kl:
                    early_stopped = True
                    break

        # バッファクリア
        self.buffer.clear()

        # LSTM状態リセット（次のrollout用）
        self.actor_critic.reset_state()

        if n_updates > 0:
            self.last_policy_loss = total_policy_loss / n_updates
            self.last_value_loss = total_value_loss / n_updates
            self.last_entropy = total_entropy / n_updates
            self.last_approx_kl = total_approx_kl / n_updates
            self.last_clip_fraction = total_clip_fraction / n_updates
            self.last_explained_var = total_explained_var / n_updates

            return {
                "policy_loss": self.last_policy_loss,
                "value_loss": self.last_value_loss,
                "entropy": self.last_entropy,
                "approx_kl": self.last_approx_kl,
                "clip_fraction": self.last_clip_fraction,
                "explained_var": self.last_explained_var,
            }
        return None

    def _update_sequence_batch(self, batch: tuple) -> dict:
        """シーケンスバッチの更新

        - burn_in部分はmaskで除外
        - LSTM状態はリセットして処理
        """
        obs, actions, old_log_probs, advantages, returns, mask = batch
        B, L = actions.shape

        # Advantage正規化（有効部分のみ）
        valid_adv = advantages[mask > 0]
        if len(valid_adv) > 0:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std

        # Forward (stateful=False)
        new_log_probs, entropy, values = self.actor_critic.evaluate_sequences(
            Variable(obs.astype(np.float32)), actions
        )

        # Policy loss (PPO Clipped) with mask
        log_ratio = new_log_probs - Variable(old_log_probs.astype(np.float32))
        ratio = F.exp(log_ratio)
        adv_v = Variable(advantages.astype(np.float32))

        surr1 = ratio * adv_v

        # Clipped ratio
        ratio_data = ratio.data_required
        clipped_ratio = np.clip(ratio_data, 1 - self.clip_eps, 1 + self.clip_eps)
        surr2 = Variable(clipped_ratio.astype(np.float32)) * adv_v

        # min(surr1, surr2)
        surr1_data = surr1.data_required
        surr2_data = surr2.data_required
        use_surr1 = (surr1_data <= surr2_data).astype(np.float32)
        policy_loss_per_step = surr1 * Variable(use_surr1) + surr2 * Variable(1 - use_surr1)

        # Apply mask
        mask_v = Variable(mask.astype(np.float32))
        masked_policy_loss = policy_loss_per_step * mask_v
        total_mask = np.sum(mask)
        policy_loss = -F.sum(masked_policy_loss) / total_mask

        # Value loss with mask
        returns_v = Variable(returns.astype(np.float32))
        value_diff = values - returns_v
        masked_value_loss = value_diff * value_diff * mask_v
        value_loss = F.sum(masked_value_loss) / total_mask

        # Entropy with mask
        masked_entropy = entropy * mask_v
        entropy_mean = F.sum(masked_entropy) / total_mask

        # Approx KL (for early stopping): E[(r-1) - log(r)]
        log_ratio_data = log_ratio.data_required
        ratio_data = np.exp(log_ratio_data)
        approx_kl = float(np.sum(((ratio_data - 1.0) - log_ratio_data) * mask) / total_mask)

        # Clip fraction: ratioがclipされた割合
        clipped = np.abs(ratio_data - 1.0) > self.clip_eps
        clip_fraction = float(np.sum(clipped * mask) / total_mask)

        # Explained variance: 1 - Var(returns - values) / Var(returns)
        values_data = values.data_required
        valid_returns = returns[mask > 0]
        valid_values = values_data[mask > 0]
        if len(valid_returns) > 1:
            var_returns = np.var(valid_returns)
            if var_returns > 1e-8:
                explained_var = 1 - np.var(valid_returns - valid_values) / var_returns
            else:
                explained_var = 0.0
        else:
            explained_var = 0.0

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

        # 更新
        self.actor_critic.cleargrads()
        loss.backward()

        # Gradient clipping
        self._clip_grad_norm()

        self.optimizer.update()

        # LSTM状態リセット（次のバッチ用）
        self.actor_critic.reset_state()

        return {
            "policy_loss": float(policy_loss.data_required),
            "value_loss": float(value_loss.data_required),
            "entropy": float(entropy_mean.data_required),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "explained_var": float(explained_var),
        }

    def _clip_grad_norm(self):
        """Gradient clipping (global norm)"""
        grads = []
        for param in self.actor_critic.params():
            if param.grad is not None:
                grads.append(param.grad.data_required.flatten())

        if not grads:
            return

        all_grads = np.concatenate(grads)
        global_norm = np.sqrt(np.sum(all_grads**2))

        if global_norm > self.max_grad_norm:
            scale = self.max_grad_norm / global_norm
            for param in self.actor_critic.params():
                if param.grad is not None:
                    param.grad.data = param.grad.data_required * scale

    def decay_epsilon(self):
        """PPOはepsilon-greedyを使わないのでno-op"""
        pass


def ppo_rnn_stats_extractor(agent) -> str | None:
    """PPO+RNN Agent用の統計抽出関数"""
    policy_loss = getattr(agent, "last_policy_loss", None)
    value_loss = getattr(agent, "last_value_loss", None)
    entropy = getattr(agent, "last_entropy", None)
    approx_kl = getattr(agent, "last_approx_kl", None)
    clip_fraction = getattr(agent, "last_clip_fraction", None)
    explained_var = getattr(agent, "last_explained_var", None)
    total_steps = getattr(agent, "last_total_steps", None)

    parts = []
    if policy_loss is not None:
        parts.append(f"π={policy_loss:.4f}")
    if value_loss is not None:
        parts.append(f"V={value_loss:.4f}")
    if entropy is not None:
        parts.append(f"H={entropy:.3f}")
    if approx_kl is not None:
        parts.append(f"KL={approx_kl:.4f}")
    if clip_fraction is not None:
        parts.append(f"Clip={clip_fraction:.3f}")
    if explained_var is not None:
        parts.append(f"ExplVar={explained_var:.3f}")
    if total_steps is not None:
        parts.append(f"Steps={total_steps}")

    if not parts:
        return None
    return ", " + ", ".join(parts)
