import numpy as np
from collections import deque
import random
from typing import Protocol
from abc import ABC, abstractmethod


class Agent(Protocol):
    """エージェントのプロトコル（離散/連続行動空間対応）"""

    def get_action(self, state: np.ndarray) -> int | np.ndarray: ...
    def act(self, state: np.ndarray, explore: bool = True) -> int | np.ndarray: ...
    def store(self, state, action, reward, next_state, done, *, terminated=None) -> None: ...
    def update(self) -> float | dict | None: ...
    def decay_epsilon(self) -> None: ...


class BaseAgent(ABC):
    """エージェントの抽象基底クラス（離散/連続行動空間対応）"""

    epsilon: float = 0.0

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int | np.ndarray:
        """探索を含むアクション選択"""
        ...

    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> int | np.ndarray:
        """アクションを選択

        Args:
            state: 観測
            explore: Trueなら探索モード、Falseなら評価モード
        """
        ...

    @abstractmethod
    def store(self, state, action, reward, next_state, done, *, terminated=None) -> None:
        """経験をバッファに保存

        Args:
            terminated: ブートストラップカット用（DRQN用、省略時はdoneを使用）
        """
        ...

    @abstractmethod
    def update(self) -> float | dict | None:
        """ネットワークを更新"""
        ...

    def decay_epsilon(self) -> None:
        """エピソード終了時にepsilonを減衰（オーバーライド可能）"""
        pass


class ReplayBuffer:
    """Experience Replay用のバッファ（離散/連続行動空間対応）"""

    def __init__(self, capacity: int = 10000, continuous_action: bool = False):
        """
        Args:
            capacity: バッファの最大サイズ
            continuous_action: Trueなら連続行動空間（action_dtype=float32）
        """
        self.buffer: deque = deque(maxlen=capacity)
        self.continuous_action = continuous_action

    def push(self, state, action, reward, next_state, done):
        """経験をバッファに追加"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """バッファからランダムにサンプリング"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        action_dtype = np.float32 if self.continuous_action else np.int64

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=action_dtype),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
