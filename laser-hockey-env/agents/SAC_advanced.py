import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from .AgentInterface import AgentInterface
from abc import ABC, abstractmethod

class SAC(AgentInterface, ABC):
    """
    Implementation of the SAC algorithm
    """
    def __init__(self, env: gym.Env, n_hidden_neurons: int = 256, alpha=0.2, gamma=0.99, tau=0.005,) -> None:
        """
        Initialize the SAC algorithm with the gym environment and the hyperparameters

        :param env: gym environment (e.g. LaserHockey-v0)
        :param n_hidden_neurons: number of neurons in the fully connected layer(s) (default: 256)
        :param alpha: temperature parameter that determines the relative importance of the entropy term against the reward (default: 0.2)
        :param gamma: discount factor for future rewards (default: 0.99)
        :param tau: target smoothing coefficient (default: 0.005)
        """
        #super(SAC, self).__init__(env=env)
        super().__init__(env=env)
        # Save the environment and the hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        # Define the neural network architecture
        n_input_neurons = env.observation_space.shape[0] # * env.observation_space.shape[1]
        print("number of input neurons: ", n_input_neurons)
        n_output_neurons = env.action_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input_neurons, n_hidden_neurons),
            nn.ReLU(),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(n_hidden_neurons, n_output_neurons)
        self.log_std_head = nn.Linear(n_hidden_neurons, n_output_neurons)

        # Action rescaling
        low, high = env.action_space.low, env.action_space.high
        self.action_scale = torch.tensor((high - low) / 2.)
        self.action_bias = torch.tensor((high + low) / 2.)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the action from the net output

        :param x: input tensor
        :return: action tensor, log probability of the action, mean of the action
        """
        # Pass the input tensor through the net to calculate the mean and log_std
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)    # log_std.exp()

        # Sample the action from a normal distribution with the just calculated mean and standard deviation
        normal = torch.distributions.Normal(loc=mean, scale=std)
        z = normal.rsample()

        # Normalize the action to be in the range of the action space
        action = torch.tanh(z)
        action = action * self.action_scale + self.action_bias
        action = action.clamp(self.env.action_space.low[0], self.env.action_space.high[0])

        # Calculate the log probability of the action
        epsilon = 1e-6  # To avoid log(0)
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - action.pow(2)) + epsilon)
        log_prob = log_prob.sum(0, keepdim=True)

        rescaled_mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, rescaled_mean

    def update(self, replay_buffer: object, batch_size: int = 256) -> None:
        """
        Update function to update the neural network weights using the SAC algorithm

        :param replay_buffer: replay buffer object to sample the transitions from
        :param batch_size: batch size (default: 256)
        """
        # Sample a batch of transitions from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        # Compute the target Q value
        with torch.no_grad():   # no_grad() prevents calculating gradients for the target network
            next_action, next_log_prob, _ = self.get_action(next_state)
            # Reshape the next_action tensor
            next_action = next_action.unsqueeze(1)

            # Calculate the target value
            target_value = self.alpha * next_log_prob - self.net(torch.cat([next_state, next_action], 1))
            target_q = reward + (1 - done) * self.gamma * target_value

        # Gradient descent step --> update the weights of the neural network
        current_q = self.net(torch.cat([state, action], 1))
        # use Mean Squared Error (MSE) loss function to calculate the loss
        q_loss = F.mse_loss(current_q, target_q)

        # Update the weights of the neural network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
