import gymnasium as gym
import numpy as np
from collections import deque
import random

import dpl.layers as L
import dpl.functions as F
from dpl import Variable
from dpl.optimizers import Adam


def mse_loss(pred: Variable, target: Variable) -> Variable:
    """Mean Squared Error loss (微分可能な実装)"""
    error = pred - target
    loss = error * error
    return F.sum(loss) / Variable(np.array(len(pred.data_required), dtype=np.float32))


class QNet(L.Sequential):
    """CartPole用のQ-Network

    入力: 状態(4次元: カート位置、カート速度、ポール角度、ポール角速度)
    出力: 各アクションのQ値(2次元: 左、右)
    """

    def __init__(self, state_size: int = 4, action_size: int = 2, hidden_size: int = 128):
        super().__init__(
            L.Linear(hidden_size),
            F.relu,
            L.Linear(hidden_size),
            F.relu,
            L.Linear(action_size),
        )
        self.state_size = state_size
        self.action_size = action_size


class ReplayBuffer:
    """Experience Replay用のバッファ"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """経験をバッファに追加"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """バッファからランダムにサンプリング"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Double DQN Agent

    - Experience Replay: 過去の経験をバッファに保存し、ミニバッチで学習
    - Target Network: 安定した学習のために定期的にコピー
    - Double DQN: action選択はqnet、Q値評価はtarget_qnetで行う
    - MSE loss
    - Gradient clipping: 勾配爆発を防ぐ
    - epsilon-greedy探索
    """

    def __init__(
        self,
        state_size: int = 4,
        action_size: int = 2,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,  # per episode
        lr: float = 5e-4,
        batch_size: int = 64,
        buffer_size: int = 10000,
        tau: float = 0.0,  # soft update coefficient (use 0 for hard update)
        target_update_freq: int = 500,  # hard update frequency
        hidden_size: int = 64,
        grad_clip: float = 1.0,
        warmup_steps: int = 500,  # warm-up before learning
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
            if isinstance(main_layer, L.Linear):
                target_layer.W.data = (
                    self.tau * main_layer.W.data + (1 - self.tau) * target_layer.W.data
                )
                if main_layer.b is not None:
                    target_layer.b.data = (
                        self.tau * main_layer.b.data + (1 - self.tau) * target_layer.b.data
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
        # 全パラメータの勾配を集める
        grads = []
        for param in self.qnet.params():
            if param.grad is not None:
                grads.append(param.grad.data_required.flatten())

        if not grads:
            return

        # Global normを計算
        all_grads = np.concatenate(grads)
        global_norm = np.sqrt(np.sum(all_grads ** 2))

        # Clipping
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
        # Warm-up: 十分な経験が貯まるまで学習しない
        if len(self.buffer) < self.warmup_steps:
            return None
        if len(self.buffer) < self.batch_size:
            return None

        # バッファからサンプリング
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Double DQN: action選択はqnet、Q値評価はtarget_qnetで行う
        next_states_var = Variable(next_states)

        # メインネットワークでaction選択
        next_q_main = self.qnet(next_states_var).data_required  # (batch, action_size)
        best_actions = np.argmax(next_q_main, axis=1)  # (batch,)

        # ターゲットネットワークでQ値評価
        next_q_target = self.target_qnet(next_states_var).data_required  # (batch, action_size)
        max_next_q = next_q_target[np.arange(len(best_actions)), best_actions]  # (batch,)

        targets = rewards + self.gamma * max_next_q * (1 - dones)  # (batch,)

        # 現在のQ値を計算（勾配あり）
        states_var = Variable(states)
        q_values = self.qnet(states_var)  # (batch, action_size)

        # 選択されたアクションのQ値を取得
        # actions: (batch,) → one-hot化して掛け合わせる
        action_masks = np.eye(self.action_size)[actions]  # (batch, action_size)
        current_q = F.sum(q_values * Variable(action_masks.astype(np.float32)), axis=1)  # (batch,)

        # MSE loss
        targets_var = Variable(targets.astype(np.float32))
        loss = mse_loss(current_q, targets_var)

        # 勾配をクリアして逆伝播
        self.qnet.cleargrads()
        loss.backward()

        # Gradient clipping (global norm)
        self._clip_grads()

        self.optimizer.update()

        # Target networkを更新
        self.learn_step += 1
        if self.tau > 0:
            # Soft update（毎ステップ）
            self._soft_update_target()
        else:
            # Hard update（一定間隔）
            if self.learn_step % self.target_update_freq == 0:
                self._hard_update_target()

        return float(loss.data_required)

    def decay_epsilon(self):
        """エピソード終了時にepsilonを減衰"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(num_episodes: int = 500, render: bool = False, eval_interval: int = 100):
    """DQNエージェントを学習

    Args:
        num_episodes: 学習エピソード数
        render: 学習中にrenderするか
        eval_interval: 評価間隔（エピソード数）
    """
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    eval_env = gym.make("CartPole-v1")  # 評価用環境
    agent = DQNAgent()

    episode_rewards = []
    episode_losses = []
    eval_returns = []  # (episode, eval_return) のリスト
    total_steps = 0  # 総ステップ数

    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0
        losses = []

        while True:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 経験を保存
            agent.store(observation, action, reward, next_observation, done)

            # Q-Networkを更新（4ステップに1回、安定化のため）
            total_steps += 1
            if total_steps % 4 == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            total_reward += reward
            observation = next_observation

            if done:
                break

        # エピソード終了時にepsilonを減衰
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        avg_loss = np.mean(losses) if losses else 0
        episode_losses.append(avg_loss)

        # 進捗を表示
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}: Reward = {total_reward:.0f}, "
                  f"Avg(10) = {avg_reward:.1f}, Loss = {avg_loss:.4f}, "
                  f"Epsilon = {agent.epsilon:.3f}")

        # 評価returnを計測
        if (episode + 1) % eval_interval == 0:
            eval_return = eval_cartpole(agent, eval_env, n=20)
            eval_returns.append((episode + 1, eval_return))
            print(f"  → Eval Return (n=20): {eval_return:.1f}")

        # 早期終了（十分に学習できた場合）
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"\nSolved in {episode + 1} episodes!")
            # 最終評価
            eval_return = eval_cartpole(agent, eval_env, n=20)
            eval_returns.append((episode + 1, eval_return))
            print(f"  → Final Eval Return (n=20): {eval_return:.1f}")
            break

    env.close()
    eval_env.close()
    return episode_rewards, episode_losses, eval_returns, agent


def eval_cartpole(agent: DQNAgent, env, n: int = 20) -> float:
    """評価return: ε=0で複数エピソード実行し平均rewardを返す

    Args:
        agent: 評価するエージェント
        env: CartPole環境
        n: 評価エピソード数

    Returns:
        平均reward
    """
    rs = []
    for _ in range(n):
        s, _ = env.reset()
        done = False
        total = 0
        while not done:
            a = agent.act(s, epsilon=0.0)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        rs.append(total)
    return float(np.mean(rs))


def evaluate_agent(agent: DQNAgent, num_episodes: int = 3):
    """学習済みエージェントをrenderして評価"""
    env = gym.make("CartPole-v1", render_mode="human")

    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0

        while True:
            # 学習済みのQ-Networkで行動選択（greedyに）
            state_var = Variable(observation.reshape(1, -1).astype(np.float32))
            q_values = agent.qnet(state_var)
            action = int(np.argmax(q_values.data_required[0]))

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Evaluation Episode {episode + 1}: Reward = {total_reward:.0f}")
                break

    env.close()


if __name__ == "__main__":
    # DQNで学習
    print("Training DQN Agent...")
    rewards, losses, eval_returns, agent = train_dqn(num_episodes=500, render=False)

    # 結果を表示
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(rewards)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("DQN: Episode Rewards")
    ax1.grid(True)

    ax2.plot(losses)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Loss")
    ax2.set_title("DQN: Average Loss per Episode")
    ax2.grid(True)

    # 評価returnをプロット
    if eval_returns:
        episodes, returns = zip(*eval_returns)
        ax3.plot(episodes, returns, marker="o", markersize=8, linewidth=2)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Eval Return")
        ax3.set_title("DQN: Eval Return (ε=0, n=20)")
        ax3.grid(True)
        ax3.axhline(y=500, color="r", linestyle="--", label="Max (500)")
        ax3.legend()

    plt.tight_layout()
    plt.show()

    # 学習済みエージェントで評価（render）
    print("\nEvaluating trained agent...")
    evaluate_agent(agent, num_episodes=3)
