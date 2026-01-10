import numpy as np
from collections import deque
import random
from typing import Protocol
from abc import ABC, abstractmethod


class Agent(Protocol):
    """エージェントのプロトコル"""

    def get_action(self, state: np.ndarray) -> int: ...
    def act(self, state: np.ndarray, epsilon: float) -> int: ...
    def store(self, state, action, reward, next_state, done, *, terminated=None) -> None: ...
    def update(self) -> float | None: ...
    def decay_epsilon(self) -> None: ...


class BaseAgent(ABC):
    """エージェントの抽象基底クラス"""

    epsilon: float = 0.0

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        """探索を含むアクション選択"""
        ...

    @abstractmethod
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """指定したepsilonでアクションを選択"""
        ...

    @abstractmethod
    def store(self, state, action, reward, next_state, done, *, terminated=None) -> None:
        """経験をバッファに保存

        Args:
            terminated: ブートストラップカット用（DRQN用、省略時はdoneを使用）
        """
        ...

    @abstractmethod
    def update(self) -> float | None:
        """ネットワークを更新"""
        ...

    def decay_epsilon(self) -> None:
        """エピソード終了時にepsilonを減衰（オーバーライド可能）"""
        pass


class ReplayBuffer:
    """Experience Replay用のバッファ"""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

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
