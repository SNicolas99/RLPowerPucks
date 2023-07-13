from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from utils.memory import ReplayBuffer
from environments import GymEnvironment


class BaseAlgorithm(ABC):
    """
    Base class for all algorithms.
    """

    @abstractmethod
    def __init__(self, env: GymEnvironment, device: str = 'cpu', **kwargs):
        """
        Initialize the algorithm.

        :param env: The environment to learn from.
        :param device: The device to use for training. (default: 'cpu')
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.env = env

        self.obs_dim, self.action_dim = self.env.obs_dim, self.env.action_dim
        self.device = device

        self.buffer = ReplayBuffer(kwargs.get('replay_buffer_size', 1000000))

    @abstractmethod
    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        Choose an action given the current state.

        :param state: The current state.
        :return: The action to take.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        """
        Update the algorithm's parameters.
        """
        raise NotImplementedError

    def store_transition(self, transition: dict[str, np.ndarray]):
        """
        Store a transition in the replay buffer.

        :param transition: The transition that should be stored.
        """
        self.buffer.store_transition(transition)

    @abstractmethod
    def save_model(self, path: str):
        """
        Save the model's parameters.

        :param path: The path to save the model to.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path: str):
        """
        Load the model's parameters.

        :param path: The path to load the model from.
        """
        raise NotImplementedError
