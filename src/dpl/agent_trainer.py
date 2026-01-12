from typing import Protocol, Any, Callable
from dataclasses import dataclass
import numpy as np
import time

from dpl.agent import Agent
from dpl.core.utils import ndarray

# 統計抽出関数の型: Agent を受け取り、ログに追加する文字列を返す
StatsExtractor = Callable[[Agent], str | None]


@dataclass
class EvalResult:
    """評価結果"""

    avg_return: float
    success_rate: float
    avg_steps: float
    avg_return_success: float
    avg_return_fail: float
    mean_abs_action: float
    n: int


# 評価統計抽出関数の型: EvalResult を受け取り、表示する文字列を返す
EvalStatsExtractor = Callable[[EvalResult], str]


class Env(Protocol):
    """環境のプロトコル（離散/連続行動空間対応）"""

    def reset(self, *, seed: int | None = None) -> ndarray: ...
    def step(self, action: int | ndarray) -> tuple[ndarray, float, bool, bool, dict]: ...


class GymEnvWrapper:
    """Gymnasium環境をAgentTrainer用にラップするクラス"""

    def __init__(self, env: Any):
        """
        Args:
            env: Gymnasium環境（gym.make()で作成したもの）
        """
        self._env = env

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """環境をリセット（infoを捨ててobsのみ返す）"""
        obs, _ = self._env.reset(seed=seed)
        return obs

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """環境を1ステップ進める"""
        return self._env.step(action)

    def close(self):
        """環境を閉じる"""
        self._env.close()

    @property
    def unwrapped(self) -> Any:
        """元のGymnasium環境を取得"""
        return self._env


@dataclass
class TrainResult:
    """トレーニング結果"""

    episode_rewards: list[float]
    episode_losses: list[float]
    # Eval-A: ε=0（完全greedy）
    eval_returns: list[tuple[int, float]]
    eval_success_rates: list[tuple[int, float]]  # (episode, success_rate)
    eval_avg_steps: list[tuple[int, float]]  # (episode, avg_steps_to_goal)
    # Eval-B: 最初burn_in stepでε>0、その後ε=0（情報収集付き）
    eval_b_success_rates: list[tuple[int, float]]  # (episode, success_rate)


class AgentTrainer:
    """エージェントのトレーニングを抽象化したクラス"""

    def __init__(
        self,
        env: Env,
        agent: Agent,
        eval_env: Env | None = None,
        num_episodes: int = 500,
        eval_interval: int = 50,
        eval_n: int = 20,
        update_every: int = 4,
        render: bool = False,
        log_interval: int = 10,
        burn_in_steps: int = 0,
        burn_in_epsilon: float = 0.2,
        stats_extractor: StatsExtractor | None = None,
        eval_stats_extractor: EvalStatsExtractor | None = None,
        eval_base_seed: int | None = None,
    ):
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.agent = agent
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.eval_n = eval_n
        self.update_every = update_every
        self.render = render
        self.log_interval = log_interval
        self.burn_in_steps = burn_in_steps
        self.burn_in_epsilon = burn_in_epsilon
        self.stats_extractor = stats_extractor
        self.eval_stats_extractor = eval_stats_extractor
        self.eval_base_seed = eval_base_seed

    def train(self) -> TrainResult:
        """トレーニングを実行"""
        episode_rewards: list[float] = []
        episode_losses: list[float] = []
        eval_returns: list[tuple[int, float]] = []
        eval_success_rates: list[tuple[int, float]] = []
        eval_avg_steps: list[tuple[int, float]] = []
        eval_b_success_rates: list[tuple[int, float]] = []
        total_steps = 0
        start_time = time.time()

        for episode in range(self.num_episodes):
            total_reward, losses, steps = self._run_episode()
            total_steps += steps

            self.agent.decay_epsilon()

            episode_rewards.append(total_reward)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            episode_losses.append(avg_loss)

            # ログ出力
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode, total_reward, episode_rewards, start_time)

            # 評価
            if (episode + 1) % self.eval_interval == 0:
                # Eval-A: ε=0（完全greedy）
                eval_result = self.evaluate(n=self.eval_n, base_seed=self.eval_base_seed)
                eval_returns.append((episode + 1, eval_result.avg_return))
                eval_success_rates.append((episode + 1, eval_result.success_rate))
                eval_avg_steps.append((episode + 1, eval_result.avg_steps))

                if self.eval_stats_extractor is not None:
                    print(f"  → Eval: {self.eval_stats_extractor(eval_result)}")
                else:
                    # デフォルト表示
                    print(
                        f"  → Eval (n={eval_result.n}): Return={eval_result.avg_return:.2f}, "
                        f"Success={eval_result.success_rate*100:.0f}%, "
                        f"AvgSteps={eval_result.avg_steps:.1f}"
                    )

                # Eval-B: 最初burn_in stepでε>0、その後ε=0（情報収集付き）
                if self.burn_in_steps > 0:
                    eval_result_b = self.evaluate(
                        n=self.eval_n,
                        burn_in_steps=self.burn_in_steps,
                        burn_in_epsilon=self.burn_in_epsilon,
                    )
                    eval_b_success_rates.append((episode + 1, eval_result_b.success_rate))
                    print(
                        f"  → Eval-B (burn_in={self.burn_in_steps}, ε={self.burn_in_epsilon}): "
                        f"Success={eval_result_b.success_rate*100:.0f}%"
                    )

        return TrainResult(
            episode_rewards, episode_losses, eval_returns,
            eval_success_rates, eval_avg_steps, eval_b_success_rates
        )

    def _run_episode(self) -> tuple[float, list[float], int]:
        """1エピソードを実行"""
        observation = self.env.reset()
        # DRQN用: LSTM状態をリセット
        if hasattr(self.agent, "reset_state"):
            self.agent.reset_state()
        # DRQN用: エピソード開始
        if hasattr(self.agent, "start_episode"):
            self.agent.start_episode()

        total_reward = 0.0
        losses: list[float] = []
        steps = 0

        while True:
            action = self.agent.get_action(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # DRQN用: terminatedを別途渡す（ブートストラップカット用）
            self.agent.store(observation, action, reward, next_observation, done, terminated=terminated)

            steps += 1
            if steps % self.update_every == 0:
                loss = self.agent.update()
                if loss is not None:
                    # dict の場合は値の平均を取る
                    if isinstance(loss, dict):
                        losses.append(float(np.mean(list(loss.values()))))
                    else:
                        losses.append(loss)

            total_reward += reward
            observation = next_observation

            if done:
                break

        # DRQN用: エピソード終了
        if hasattr(self.agent, "end_episode"):
            self.agent.end_episode()

        return total_reward, losses, steps

    def _format_time(self, seconds: float) -> str:
        """秒数を読みやすい形式に変換"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        else:
            h, remainder = divmod(int(seconds), 3600)
            m, s = divmod(remainder, 60)
            return f"{h}h{m:02d}m"

    def _log_progress(
        self,
        episode: int,
        total_reward: float,
        episode_rewards: list[float],
        start_time: float,
    ):
        """進捗をログ出力"""
        avg_reward = np.mean(episode_rewards[-self.log_interval :])

        # 残り時間の計算
        elapsed = time.time() - start_time
        progress = (episode + 1) / self.num_episodes
        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            remaining = 0

        eta_str = f"ETA={self._format_time(remaining)}"

        # 外部から注入された統計抽出関数を使用
        extra_stats = ""
        if self.stats_extractor is not None:
            extracted = self.stats_extractor(self.agent)
            if extracted:
                extra_stats = extracted

        print(
            f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
            f"Avg({self.log_interval}) = {avg_reward:.2f}{extra_stats}, {eta_str}"
        )

    def evaluate(
        self,
        n: int = 20,
        burn_in_steps: int = 0,
        burn_in_epsilon: float = 0.0,
        base_seed: int | None = None,
    ) -> EvalResult:
        """評価を実行

        Args:
            n: 評価エピソード数
            burn_in_steps: 最初の数ステップで使うepsilon適用ステップ数（情報収集用）
            burn_in_epsilon: burn_in_steps中に使うepsilon値
            base_seed: 乱数シードのベース値（指定時は各エピソードで base_seed + i を使用）

        Returns:
            EvalResult: 評価結果
        """
        returns = []
        returns_success = []
        returns_fail = []
        steps_to_goal = []
        all_actions: list[float] = []

        for i in range(n):
            seed = base_seed + i if base_seed is not None else None
            s = self.eval_env.reset(seed=seed)
            # DRQN用: LSTM状態をリセット
            if hasattr(self.agent, "reset_state"):
                self.agent.reset_state()
            done = False
            total = 0.0
            steps = 0
            while not done:
                # burn_in_steps中は探索あり、その後は探索なし
                if steps < burn_in_steps:
                    explore = True
                else:
                    explore = False
                a = self.agent.act(s, explore=explore)
                # 行動を記録（連続/離散両対応）
                if isinstance(a, np.ndarray):
                    all_actions.extend(np.abs(a).flatten().tolist())
                else:
                    all_actions.append(abs(float(a)))
                s, r, terminated, truncated, _ = self.eval_env.step(a)
                done = terminated or truncated
                total += r
                steps += 1

            returns.append(total)
            if terminated:  # ゴール到達
                returns_success.append(total)
                steps_to_goal.append(steps)
            else:
                returns_fail.append(total)

        return EvalResult(
            avg_return=float(np.mean(returns)),
            success_rate=len(returns_success) / n,
            avg_steps=float(np.mean(steps_to_goal)) if steps_to_goal else 0.0,
            avg_return_success=float(np.mean(returns_success)) if returns_success else 0.0,
            avg_return_fail=float(np.mean(returns_fail)) if returns_fail else 0.0,
            mean_abs_action=float(np.mean(all_actions)) if all_actions else 0.0,
            n=n,
        )
