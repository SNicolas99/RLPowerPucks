"""
    Code is based on DDPG implementation from RL Lecture given by Georg Martius in 2023
"""


import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

import sys

sys.path.append("../../..")  # TODO: adjust path

from tools.memory import ReplayBuffer
from tools.feedforward import Feedforward, to_torch

torch.set_num_threads(4)


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class QFunction(Feedforward):
    def __init__(
        self, observation_dim, action_dim, hidden_sizes=[100, 100], learning_rate=0.0002
    ):
        super().__init__(
            input_size=observation_dim + action_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, eps=0.000001
        )
        self.loss = nn.SmoothL1Loss()

    def fit(
        self, observations, actions, targets
    ):  # all arguments should be torch tensors
        self.train()  # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations, actions)
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations, actions]))


class OUNoise:
    def __init__(self, shape, theta: float = 0.05, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (-self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


class TD3Agent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible "
                "with {}. (Require: Box)".format(observation_space, self)
            )
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Action space {} incompatible with {}."
                " (Require Box)".format(action_space, self)
            )

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.2,  # Epsilon: noise strength to add to policy # 0.1
            "discount": 0.97,  # 0.95
            "buffer_size": int(1e6),  # 1e6
            "batch_size": 128,  # 128
            "learning_rate_actor": 0.00001,  # 0.00001
            "learning_rate_critic": 0.0001,  # 0.0001
            "hidden_sizes_actor": [256, 256],  # [256, 256]
            "hidden_sizes_critic": [256, 256, 256],  # [256, 256, 256]
            "tau": 0.0002,  # 0.0002
            "hard_update_frequency": 100,  # 100
            "policy_target_update_interval": 2,  # 2
            "target_action_noise": 0.1,  # 0.2
            "target_action_noise_clip": 0.25,  # 0.5
            "use_second_critic": True,
            "use_prioritized_replay": False,
        }
        self._config.update(userconfig)
        self._eps = self._config["eps"]

        self.action_noise = OUNoise((self._action_n))

        self.buffer = ReplayBuffer(
            buffer_size=self._config["buffer_size"],
            prioritized_replay=self._config["use_prioritized_replay"],
        )

        # Q Networks
        self.Q = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
        )

        self.Q2 = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
        )

        # target Q Networks
        self.Q_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0,
        )

        self.Q2_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0,
        )

        self.policy = Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=torch.nn.Tanh(),
        )
        self.policy_target = Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=torch.nn.Tanh(),
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self._config["learning_rate_actor"],
            eps=0.000001,
        )

        self._copy_nets()

        # disable gradients for target networks
        for p in self.Q_target.parameters():
            p.requires_grad = False
        for p in self.Q2_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.train_iter = 0

    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def _update_target_nets(self):
        # update Q
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self._config["tau"] * param + (1 - self._config["tau"]) * target_param
            )
        # update Q2
        for target_param, param in zip(
            self.Q2_target.parameters(), self.Q2.parameters()
        ):
            target_param.data.copy_(
                self._config["tau"] * param + (1 - self._config["tau"]) * target_param
            )
        # update policy
        for target_param, param in zip(
            self.policy_target.parameters(), self.policy.parameters()
        ):
            target_param.data.copy_(
                self._config["tau"] * param + (1 - self._config["tau"]) * target_param
            )

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        observation = np.atleast_2d(observation)

        self.policy.eval()
        action = self.policy.predict(
            to_torch(observation)
        ).squeeze()  # action in -1 to 1 (+ noise)
        self.policy.train()

        action = action + eps * self.action_noise()

        action = self._action_space.low + (action + 1.0) / 2.0 * (
            self._action_space.high - self._action_space.low
        )
        return action

    def store_transition(self, transition):
        self.buffer.push(*transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def get_target_action_noise(self):
        noise = np.random.randn((self._action_n)) * self._config["target_action_noise"]
        return np.clip(
            noise,
            -self._config["target_action_noise_clip"],
            self._config["target_action_noise_clip"],
        )

    def optimize_actor(self, s):
        for p in self.Q.parameters():
            p.requires_grad = False
        self.Q.eval()

        self.optimizer.zero_grad()
        q = self.Q.Q_value(s, self.policy.forward(s))
        actor_loss = -torch.mean(q)
        actor_loss.backward()
        self.optimizer.step()

        for p in self.Q.parameters():
            p.requires_grad = True
        self.Q.train()

        return actor_loss.item()

    def train(self, iter_fit=32):
        fit_losses = []
        actor_losses = []
        self.train_iter += 1

        if self.train_iter % self._config["hard_update_frequency"] == 0:
            self._copy_nets()

        for i in range(iter_fit):
            # sample from the replay buffer
            s, a, r, s_prime, done = self.buffer.sample(
                batch_size=self._config["batch_size"]
            )

            s = to_torch(s)  # batch_size x obs_dim
            a = to_torch(a)  # batch_size x action_dim
            r = to_torch(r)  # batch_size x 1
            s_prime = to_torch(s_prime)  # batch_size x obs_dim
            done = to_torch(done)  # batch_size x 1

            target_action = self.policy_target.forward(s_prime)
            target_action_noise = self.get_target_action_noise()
            target_action = to_torch(
                np.clip(
                    target_action + target_action_noise,
                    self._action_space.low,
                    self._action_space.high,
                )
            )

            q_prime = self.Q_target.Q_value(s_prime, target_action)
            if self._config["use_second_critic"]:
                q_prime2 = self.Q2_target.Q_value(s_prime, target_action)
                q_prime = torch.minimum(q_prime, q_prime2)

            # target
            gamma = self._config["discount"]
            td_target = r + gamma * (1.0 - done) * q_prime

            # optimize the Q objective
            fit_loss = self.Q.fit(s, a, td_target)
            fit_losses.append(fit_loss)

            # optimize the Q2 objective
            if self._config["use_second_critic"]:
                fit_loss2 = self.Q2.fit(s, a, td_target)
                fit_losses.append(fit_loss2)

            if self.train_iter % self._config["policy_target_update_interval"] == 0:
                self._update_target_nets()

                # optimize the actor objective
                actor_loss = self.optimize_actor(s)
                actor_losses.append(actor_loss)

        return fit_losses, actor_losses


def main():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-e",
        "--env",
        action="store",
        type="string",
        dest="env_name",
        default="Pendulum-v1",
        help="Environment (default %default)",
    )
    optParser.add_option(
        "-n",
        "--eps",
        action="store",
        type="float",
        dest="eps",
        default=0.1,
        help="Policy noise (default %default)",
    )
    optParser.add_option(
        "-t",
        "--train",
        action="store",
        type="int",
        dest="train",
        default=32,
        help="number of training batches per episode (default %default)",
    )
    optParser.add_option(
        "-l",
        "--lr",
        action="store",
        type="float",
        dest="lr",
        default=0.0001,
        help="learning rate for actor/policy (default %default)",
    )
    optParser.add_option(
        "-m",
        "--maxepisodes",
        action="store",
        type="float",
        dest="max_episodes",
        default=2000,
        help="number of episodes (default %default)",
    )
    optParser.add_option(
        "-u",
        "--update",
        action="store",
        type="float",
        dest="update_every",
        default=100,
        help="number of episodes between target network updates (default %default)",
    )
    optParser.add_option(
        "-s",
        "--seed",
        action="store",
        type="int",
        dest="seed",
        default=None,
        help="random seed (default %default)",
    )
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous=True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20  # print avg reward in the interval
    max_episodes = opts.max_episodes  # max training episodes
    max_timesteps = 2000  # max timesteps in one episode

    train_iter = opts.train  # update networks for given batched after every episode
    eps = opts.eps  # noise of DDPG policy
    lr = opts.lr  # learning rate of DDPG policy
    random_seed = opts.seed
    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    agent = TD3Agent(
        env.observation_space,
        env.action_space,
        eps=eps,
        learning_rate_actor=lr,
        update_target_every=opts.update_every,
    )

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(
            f"./results/DDPG_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl",
            "wb",
        ) as f:
            pickle.dump(
                {
                    "rewards": rewards,
                    "lengths": lengths,
                    "eps": eps,
                    "train": train_iter,
                    "lr": lr,
                    "update_every": opts.update_every,
                    "losses": losses,
                },
                f,
            )

    # training loop
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        agent.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = agent.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward += reward
            agent.store_transition((ob, a, reward, ob_new, done))
            ob = ob_new
            if done or trunc:
                break

        losses.extend(agent.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(
                agent.state(),
                f"./results/DDPG_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth",
            )
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print(
                "Episode {} \t avg length: {} \t reward: {}".format(
                    i_episode, avg_length, avg_reward
                )
            )
    save_statistics()


if __name__ == "__main__":
    main()
