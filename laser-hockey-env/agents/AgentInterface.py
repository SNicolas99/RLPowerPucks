import torch
import gymnasium as gym
from abc import ABC, abstractmethod

class AgentInterface(torch.nn.Module, ABC):
    """
    Base class for RL algorithms
    """
    def __init__(self, env: gym.Env) -> None:
        """
        
        """
        super(AgentInterface, self).__init__()
        self.env = env

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(self, replay_buffer, batch_size):
        raise NotImplementedError
