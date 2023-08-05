import numpy as np
import random

class ReplayBuffer:
    def __init__(self, buffer_size, prioritized_replay=False):
        self.buffer_size = buffer_size
        self.init_weight = 1e8
        self.current_index = 0
        self.size = 0
        self.prioritized_replay = prioritized_replay
        self.last_batch_inds = None

        self.buffer = np.full(buffer_size, None, dtype=object)
        self.weights = None
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

        # get state, action, reward, next_state, done from batch as np arrays
        state = np.array([elem["state"] for elem in batch], dtype=np.float32)
        action = np.array([elem["action"] for elem in batch], dtype=np.float32)
        reward = np.array([elem["reward"] for elem in batch], dtype=np.float32)
        next_state = np.array([elem["next_state"] for elem in batch], dtype=np.float32)
        done = np.array([elem["done"] for elem in batch], dtype=np.float32)

        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)

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
                inds = np.random.choice(self.size, size=shape, p=probs, replace=False)
            return inds
        else:
            # efficient sampling of random indices
            if isinstance(shape, int):
                shape = (shape,)
            return (np.random.rand(*shape) * self.size).astype(int)

    def save(self, path):
        state = {
            "buffer": self.buffer,
            "buffer_size": self.buffer_size,
            "weights": self.weights,
            "current_index": self.current_index,
            "size": self.size,
            "prioritized_replay": self.prioritized_replay,
            "last_batch_inds": self.last_batch_inds,
        }
        np.save(state, path)
    
    def load(self, path):
        state = np.load(path)
        self.buffer = state["buffer"]
        self.buffer_size = state["buffer_size"]
        self.weights = state["weights"]
        self.current_index = state["current_index"]
        self.size = state["size"]
        self.prioritized_replay = state["prioritized_replay"]
        self.last_batch_inds = state["last_batch_inds"]
