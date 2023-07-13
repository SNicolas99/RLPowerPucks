import torch
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

class ReplayBufferPrioritized:
    def __init__(self, buffer_size, prioritized_replay=True):
        self.buffer_size = buffer_size
        self.init_weight = 1e8 # infinitely large weight for new transitions
        self.current_index = 0
        self.size = 0
        self.prioritized_replay = prioritized_replay
        self.last_batch_inds = None

        self.buffer = np.full(buffer_size, None, dtype=object)
        if self.prioritized_replay:
            self.weights = np.full(buffer_size, self.init_weight, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        elem_dict = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.size = min(self.size + 1, self.buffer_size)
        self.buffer[self.current_index] = elem_dict
        self.current_index = (self.current_index + 1) % self.buffer_size

    def sample(self, inds=None, batch_size=1):

        if self.size < batch_size:
            batch_size = self.size

        if inds is None:
            inds=self.sample_inds((batch_size, ))

        batch = self.buffer[inds]

        if self.prioritized_replay:
            self.last_batch_inds = inds

        # hier vllt das torch zeug rausnehmen und in der agent klasse zu torch tensors casten?
        # get state, action, reward, next_state, done from batch as torch tensors
        state = torch.tensor(np.array([elem["state"] for elem in batch]), dtype=torch.float32)
        action = torch.tensor(np.array([elem["action"] for elem in batch]), dtype=torch.float32)
        reward = torch.tensor(np.array([elem["reward"] for elem in batch]), dtype=torch.float32)
        next_state = torch.tensor(
            np.array([elem["next_state"] for elem in batch]), dtype=torch.float32
        )
        done = torch.tensor(np.array([elem["done"] for elem in batch], dtype=np.float32))

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def get_last_probs(self):
        if self.prioritized_replay:
            probs = self.weights[self.last_batch_inds]
            return probs / probs.sum()
        else:
            return None

    def update_priorities(self, priorities):
        if self.prioritized_replay:
            self.weights[self.last_batch_inds] = priorities
            self.last_batch_inds = None
        else:
            raise ValueError("Replay buffer does not use prioritized replay")

    def sample_inds(self, shape):
        if self.prioritized_replay:
            # sample indices with probability proportional to their weight
            weights = self.weights[: self.size]
            probs = weights / weights.sum()
            if np.product(shape) > self.size:
                inds = np.random.choice(self.size, size=shape, p=probs, replace=True)
            else:
                # mit replace vllt sogar generell besser?
                inds = np.random.choice(self.size, size=shape, p=probs, replace=False)
            return inds
        else:
            # efficient sampling of random indices
            if isinstance(shape, int):
                shape = (shape,)
            return (np.random.rand(*shape) * self.size).astype(int)

