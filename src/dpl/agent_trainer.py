from typing import Protocol, Any
from dataclasses import dataclass
import numpy as np

from dpl.agent import Agent


class Env(Protocol):
    """環境のプロトコル（離散/連続行動空間対応）"""

    def reset(self) -> np.ndarray: ...
    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]: ...


class GymEnvWrapper:
    """Gymnasium環境をAgentTrainer用にラップするクラス"""

    def __init__(self, env: Any):
        """
        Args:
            env: Gymnasium環境（gym.make()で作成したもの）
        """
        self._env = env

    def reset(self) -> np.ndarray:
        """環境をリセット（infoを捨ててobsのみ返す）"""
        obs, _ = self._env.reset()
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
        eval_env: Env,
        agent: Agent,
        num_episodes: int = 500,
        eval_interval: int = 50,
        eval_n: int = 20,
        update_every: int = 4,
        render: bool = False,
        log_interval: int = 10,
        burn_in_steps: int = 0,
        burn_in_epsilon: float = 0.2,
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.eval_n = eval_n
        self.update_every = update_every
        self.render = render
        self.log_interval = log_interval
        self.burn_in_steps = burn_in_steps
        self.burn_in_epsilon = burn_in_epsilon

    def train(self) -> TrainResult:
        """トレーニングを実行"""
        episode_rewards: list[float] = []
        episode_losses: list[float] = []
        eval_returns: list[tuple[int, float]] = []
        eval_success_rates: list[tuple[int, float]] = []
        eval_avg_steps: list[tuple[int, float]] = []
        eval_b_success_rates: list[tuple[int, float]] = []
        total_steps = 0

        for episode in range(self.num_episodes):
            total_reward, losses, steps = self._run_episode()
            total_steps += steps

            self.agent.decay_epsilon()

            episode_rewards.append(total_reward)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            episode_losses.append(avg_loss)

            # ログ出力
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode, total_reward, episode_rewards, avg_loss)

            # 評価
            if (episode + 1) % self.eval_interval == 0:
                # Eval-A: ε=0（完全greedy）
                eval_return, success_rate, avg_steps, avg_success, avg_fail, mean_abs_action = self.evaluate(n=self.eval_n)
                eval_returns.append((episode + 1, eval_return))
                eval_success_rates.append((episode + 1, success_rate))
                eval_avg_steps.append((episode + 1, avg_steps))
                print(
                    f"  → Eval-A (ε=0, n={self.eval_n}): Return={eval_return:.2f}, "
                    f"Success={success_rate*100:.0f}% ({int(success_rate*self.eval_n)}/{self.eval_n}), "
                    f"AvgSteps={avg_steps:.1f}, Mean|Action|={mean_abs_action:.3f}"
                )

                # Eval-B: 最初burn_in stepでε>0、その後ε=0（情報収集付き）
                if self.burn_in_steps > 0:
                    _, success_rate_b, _, _, _, _ = self.evaluate(
                        n=self.eval_n,
                        burn_in_steps=self.burn_in_steps,
                        burn_in_epsilon=self.burn_in_epsilon,
                    )
                    eval_b_success_rates.append((episode + 1, success_rate_b))
                    print(
                        f"  → Eval-B (burn_in={self.burn_in_steps}, ε={self.burn_in_epsilon}, n={self.eval_n}): "
                        f"Success={success_rate_b*100:.0f}% ({int(success_rate_b*self.eval_n)}/{self.eval_n})"
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

    def _log_progress(
        self,
        episode: int,
        total_reward: float,
        episode_rewards: list[float],
        avg_loss: float,
    ):
        """進捗をログ出力"""
        avg_reward = np.mean(episode_rewards[-self.log_interval :])
        epsilon = getattr(self.agent, "epsilon", 0.0)

        # Actor/Critic lossがあれば表示
        actor_loss = getattr(self.agent, "last_actor_loss", None)
        critic_loss = getattr(self.agent, "last_critic_loss", None)

        loss_str = f"Loss = {avg_loss:.4f}"
        if actor_loss is not None and critic_loss is not None:
            loss_str = f"ActorLoss = {actor_loss:.4f}, CriticLoss = {critic_loss:.4f}"

        # 1) V(s) の統計
        value_mean = getattr(self.agent, "last_value_mean", None)
        value_std = getattr(self.agent, "last_value_std", None)
        value_min = getattr(self.agent, "last_value_min", None)
        value_max = getattr(self.agent, "last_value_max", None)
        target_mean = getattr(self.agent, "last_target_mean", None)
        target_std = getattr(self.agent, "last_target_std", None)

        # 2) Advantage の統計（正規化前）
        adv_mean = getattr(self.agent, "last_adv_mean_raw", None)
        adv_std = getattr(self.agent, "last_adv_std_raw", None)
        adv_min = getattr(self.agent, "last_adv_min_raw", None)
        adv_max = getattr(self.agent, "last_adv_max_raw", None)

        # 3) 方策の std の統計
        policy_std_mean = getattr(self.agent, "last_policy_std_mean", None)
        policy_std_min = getattr(self.agent, "last_policy_std_min", None)
        policy_std_max = getattr(self.agent, "last_policy_std_max", None)

        # 4) States の多様性チェック
        states_shape = getattr(self.agent, "last_states_shape", None)
        states_std = getattr(self.agent, "last_states_std_per_dim", None)

        debug_str = ""
        if value_mean is not None:
            debug_str += f"\n  → V(s): mean={value_mean:.1f} std={value_std:.1f} [{value_min:.1f}, {value_max:.1f}]"
            debug_str += f" | Target: mean={target_mean:.1f} std={target_std:.1f}"
        if adv_mean is not None:
            debug_str += f"\n  → Adv(raw): mean={adv_mean:.2f} std={adv_std:.2f} [{adv_min:.2f}, {adv_max:.2f}]"
        if policy_std_mean is not None:
            debug_str += f"\n  → Policy σ: mean={policy_std_mean:.3f} [{policy_std_min:.3f}, {policy_std_max:.3f}]"
        if states_shape is not None:
            std_str = ", ".join([f"{s:.3f}" for s in states_std])
            debug_str += f"\n  → States: shape={states_shape}, std=[{std_str}]"

        print(
            f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
            f"Avg({self.log_interval}) = {avg_reward:.2f}, {loss_str}, "
            f"Epsilon = {epsilon:.3f}{debug_str}"
        )

    def evaluate(
        self,
        n: int = 20,
        burn_in_steps: int = 0,
        burn_in_epsilon: float = 0.0,
    ) -> tuple[float, float, float, float, float, float]:
        """評価を実行

        Args:
            n: 評価エピソード数
            burn_in_steps: 最初の数ステップで使うepsilon適用ステップ数（情報収集用）
            burn_in_epsilon: burn_in_steps中に使うepsilon値

        Returns:
            (avg_return, success_rate, avg_steps_to_goal, avg_return_success, avg_return_fail, mean_abs_action)
            - avg_return: 平均リターン
            - success_rate: ゴール到達率 (0.0 ~ 1.0)
            - avg_steps_to_goal: ゴール到達時の平均ステップ数（到達なしの場合は0）
            - avg_return_success: 成功時の平均リターン（成功なしの場合は0）
            - avg_return_fail: 失敗時の平均リターン（失敗なしの場合は0）
            - mean_abs_action: 平均絶対行動値
        """
        returns = []
        returns_success = []
        returns_fail = []
        steps_to_goal = []
        all_actions: list[float] = []

        for _ in range(n):
            s = self.eval_env.reset()
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

        avg_return = float(np.mean(returns))
        success_rate = len(returns_success) / n
        avg_steps = float(np.mean(steps_to_goal)) if steps_to_goal else 0.0
        avg_return_success = float(np.mean(returns_success)) if returns_success else 0.0
        avg_return_fail = float(np.mean(returns_fail)) if returns_fail else 0.0
        mean_abs_action = float(np.mean(all_actions)) if all_actions else 0.0

        return avg_return, success_rate, avg_steps, avg_return_success, avg_return_fail, mean_abs_action
