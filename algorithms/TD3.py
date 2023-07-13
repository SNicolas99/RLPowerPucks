from algorithms import BaseAlgorithm
import numpy as np
from environments import GymEnvironment


class TD3(BaseAlgorithm):
    """TD3 algorithm implementation."""
    def __init__(self, env: GymEnvironment, device: str = 'cpu', **kwargs):
        """
        Implementation of the Twin Delayed Deep Deterministic Policy Gradient algorithm.

        :param env: The gym environment to learn from.
        :param device: The device to use for training. (default: 'cpu')
        :param kwargs: Additional arguments.
        """
        super().__init__(env=env, device=device, **kwargs)

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        Choose an action given the current state.

        :param state: The current state.
        :return: The action to take.
        """
        raise NotImplementedError

    def learn(self):
        """
        Update the algorithm's parameters.
        """
        raise NotImplementedError

    def save_model(self, path: str):
        """
        Save the model's parameters.

        :param path: The path to save the model to.
        """
        raise NotImplementedError

    def load_model(self, path: str):
        """
        Load the model's parameters.

        :param path: The path to load the model from.
        """
        raise NotImplementedError