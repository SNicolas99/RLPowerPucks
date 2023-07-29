import torch
import numpy as np


class Scaler:
    def __init__(self, env, hockey=False):
        self.env = env
        action_space = env.action_space

        if hockey:
            self.action_low = torch.tensor(action_space.low[:4], dtype=torch.float32, requires_grad=False)
            self.action_high = torch.tensor(action_space.high[:4], dtype=torch.float32, requires_grad=False)
            self.action_range = self.action_high - self.action_low
        else:
            self.action_low = torch.tensor(action_space.low, dtype=torch.float32, requires_grad=False)
            self.action_high = torch.tensor(action_space.high, dtype=torch.float32, requires_grad=False)
            self.action_range = self.action_high - self.action_low

        self.observation_space = env.observation_space
        self.observation_low = torch.tensor(
            self.observation_space.low, dtype=torch.float32, requires_grad=False
        )
        self.observation_high = torch.tensor(
            self.observation_space.high, dtype=torch.float32, requires_grad=False
        )
        self.observation_range = self.observation_high - self.observation_low

        if self.action_low is None or self.action_high is None or np.isinf(self.action_range).any():
            self.action_scaling = False
        else:
            self.action_scaling = True
            
        if self.observation_low is None or self.observation_high is None or np.isinf(self.observation_range).any():
            self.observation_scaling = False
        else:
            self.observation_scaling = True

    def scale_action(self, action):
        if not self.action_scaling:
            return action
        return self.action_low + (action + 1.0) * 0.5 * self.action_range

    def unscale_action(self, action):
        if not self.action_scaling:
            return action
        return ((action - self.action_low) / self.action_range)*2 - 1.0

    def scale_state(self, state):
        if not self.observation_scaling:
            return state
        return self.observation_low + (state + 1.0) * 0.5 * self.observation_range

    def unscale_state(self, state):
        if not self.observation_scaling:
            return state
        return ((state - self.observation_low) / self.observation_range)*2 - 1.0
    
    def save(self, filename):
        torch.save(
            {
                "action_low": self.action_low,
                "action_high": self.action_high,
                "action_range": self.action_range,
                "observation_low": self.observation_low,
                "observation_high": self.observation_high,
                "observation_range": self.observation_range,
            },
            filename,
        )
    
    def load(self, filename):
        data = torch.load(filename)
        self.action_low = data["action_low"]
        self.action_high = data["action_high"]
        self.action_range = data["action_range"]
        self.observation_low = data["observation_low"]
        self.observation_high = data["observation_high"]
        self.observation_range = data["observation_range"]
