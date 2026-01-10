from typing import Protocol
from dataclasses import dataclass
import numpy as np

from dpl.agent import Agent


class Env(Protocol):
    """環境のプロトコル"""

    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...


@dataclass
class TrainResult:
    """トレーニング結果"""

    episode_rewards: list[float]
    episode_losses: list[float]
    eval_returns: list[tuple[int, float]]


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

    def train(self) -> TrainResult:
        """トレーニングを実行"""
        episode_rewards: list[float] = []
        episode_losses: list[float] = []
        eval_returns: list[tuple[int, float]] = []
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
                eval_return = self.evaluate(n=self.eval_n)
                eval_returns.append((episode + 1, eval_return))
                print(f"  → Eval Return (n={self.eval_n}): {eval_return:.2f}")

        return TrainResult(episode_rewards, episode_losses, eval_returns)

    def _run_episode(self) -> tuple[float, list[float], int]:
        """1エピソードを実行"""
        observation = self.env.reset()
        total_reward = 0.0
        losses: list[float] = []
        steps = 0

        while True:
            action = self.agent.get_action(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.agent.store(observation, action, reward, next_observation, done)

            steps += 1
            if steps % self.update_every == 0:
                loss = self.agent.update()
                if loss is not None:
                    losses.append(loss)

            total_reward += reward
            observation = next_observation

            if done:
                break

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
        print(
            f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
            f"Avg({self.log_interval}) = {avg_reward:.2f}, Loss = {avg_loss:.4f}, "
            f"Epsilon = {epsilon:.3f}"
        )

    def evaluate(self, n: int = 20) -> float:
        """評価を実行（ε=0で複数エピソード）"""
        rs = []
        for _ in range(n):
            s = self.eval_env.reset()
            done = False
            total = 0.0
            while not done:
                a = self.agent.act(s, epsilon=0.0)
                s, r, terminated, truncated, _ = self.eval_env.step(a)
                done = terminated or truncated
                total += r
            rs.append(total)
        return float(np.mean(rs))
