import numpy as np
import random



class ReplayBuffer:
    """
    Class for a replay buffer. Supports storing transitions and sampling batches from the buffer.
    """
    def __init__(self, buffer_size=1000000):
        """
        Initialize the replay buffer.

        :param buffer_size: The maximum size of the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = []

    def store_transition(self, transition):
        """
        Store a transition in the replay buffer.

        :param transition: The transition to store.
        """
        self.buffer.append(transition)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_from_buffer(self, batch_size) -> list[dict[str, np.ndarray]]:
        """
        Sample a batch of transitions from the replay buffer.

        :param batch_size: The size of the batch to sample.
        :return: A batch of transitions.
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        """
        Replaces the len-operator for the buffer. Returns the length of the buffer.

        :return: The number of transitions in the buffer.
        """
        return len(self.buffer)

    def save(self, path: str):
        """
        Save the replay buffer to a file.

        :param path: The path to save the buffer to.
        """
        np.save(path, self.buffer)

    def load(self, path: str):
        """
        Load a replay buffer from a file.

        :param path: The path to load the buffer from.
        """
        self.buffer = np.load(path, allow_pickle=True).tolist()
