import numpy as np

# IMPORTANT NOTICE: Partly taken from exercises from the Reinforcement Learning Course at the University of Tübingen

# class to store transitions
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(self.size, size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]

    # TODO maybe another function to update the priorities

class PER_Memory:
    def __init__(self, max_size=100000, alpha=0.6, beta=0.4):
        self.transitions = np.asarray([])
        self.priorities = np.zeros(max_size)
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta

    def add_transition(self, transitions_new, priority=None):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
        if priority is None:
            priority = self.max_priority()
        self.update_priority(self.current_idx, priority)
        self.size += 1
        self.current_idx = (self.current_idx + 1) % self.max_size

    def max_priority(self):
        return self.priorities.max() if self.size > 0 else 1.0

    def update_priority(self, idx, priority):
        self.priorities[idx] = priority

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size

        # Calculate probabilities for sampling based on priorities
        priority_probs = self.priorities[:self.size] ** self.alpha
        # Normalization
        priority_probs /= priority_probs.sum()

        # Sample indices based on probabilities
        self.inds = np.random.choice(self.size, size=batch, p=priority_probs)

        # Calculate importance sampling weights
        weights = (self.size * priority_probs[self.inds]) ** (-self.beta)
        normalized_weights = weights / weights.max()

        return self.transitions[self.inds,:], normalized_weights, self.inds

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.update_priority(idx, priority)

    def get_all_transitions(self):
        return self.transitions[:self.size]

