import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from .AgentInterface import AgentInterface
from abc import ABC, abstractmethod

class PPO(AgentInterface, ABC):
    """
    Implementation of the Proximal Policy Optimization (PPO) algorithm
    """
    def __init__(self, env, n_hidden_neurons=256, epsilon=0.2, gamma=0.99, lr=0.0003, value_coef=0.5, entropy_coef=0.01) -> None:
        super(PPO, self).__init__(env)
        self.epsilon = epsilon
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Define the neural network architecture
        n_input_neurons = env.observation_space.shape[0]
        n_output_neurons = env.action_space.shape[0]
        self.policy_net = nn.Sequential(
            nn.Linear(n_input_neurons, n_hidden_neurons),
            nn.ReLU(),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.ReLU()
        )
        self.value_net = nn.Sequential(
            nn.Linear(n_input_neurons, n_hidden_neurons),
            nn.ReLU(),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.ReLU(),
            nn.Linear(n_hidden_neurons, 1)
        )

        # Action rescaling
        low, high = env.action_space.low, env.action_space.high
        self.action_scale = torch.tensor((high - low) / 2.)
        self.action_bias = torch.tensor((high + low) / 2.)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        mean = self.policy_net(state)
        value = self.value_net(state)

        # Sample the action from a normal distribution
        normal = torch.distributions.Normal(loc=mean, scale=self.epsilon)
        action = normal.sample()

        # Normalize the action to be in the range of the action space
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        action = action.clamp(self.env.action_space.low[0], self.env.action_space.high[0])

        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def update(self, replay_buffer, batch_size):
        state, action, old_log_prob, old_value, return_, adv = replay_buffer.sample(batch_size)

        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        old_log_prob = torch.FloatTensor(old_log_prob)
        old_value = torch.FloatTensor(old_value)
        return_ = torch.FloatTensor(return_)
        adv = torch.FloatTensor(adv)

        for _ in range(10):  # Number of optimization steps
            # Calculate current log probabilities and values
            mean = self.policy_net(state)
            value = self.value_net(state)

            # Calculate action log probabilities
            normal = torch.distributions.Normal(loc=mean, scale=self.epsilon)
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

            # Calculate ratios and surrogate objective
            ratio = (log_prob - old_log_prob).exp()
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv)

            # Calculate value loss
            value_loss = F.mse_loss(value, return_)

            # Calculate entropy loss
            entropy_loss = normal.entropy().mean()

            # Calculate total loss
            loss = -surrogate.mean() + self.value_coef * value_loss - self.entropy_coef * entropy_loss

            # Update the policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
