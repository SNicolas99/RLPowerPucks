from algorithms import BaseAlgorithm
import numpy as np


class DDQN(BaseAlgorithm):
    """DDQN algorithm implementation."""
    def __init__(self, **kwargs):
        """
        Implementation of the Double Deep Q-Network algorithm.

        :param kwargs: Additional arguments.
        """
        super().__init__(**kwargs)

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