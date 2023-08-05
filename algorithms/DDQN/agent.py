import copy

import numpy as np
from q_dueling_network import QFunction
import torch as T
from memory import Memory, PER_Memory
import os
import time


## Took parts from exercise
class DQNAgent():

    def __init__(self, observation_space, action_space, logger, config):

        self.observation_dim = observation_space
        self.action_space = action_space


        self.config = config
        self.e = self.config['epsilon']
        self.PER = self.config['PER']

        if self.PER:
            self.buffer = PER_Memory(max_size=self.config["buffer_size"], beta=self.config['beta'], alpha=self.config['alpha'])
        else:
            self.buffer = Memory(max_size=self.config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self.observation_dim,
                           action_dim=self.action_space,
                           lr = self.config['learning_rate'],
                           device=self.config['device'])

        # Define the target network as a copy of the original network
        # While training, the target network will be updated with the parameters of the
        # original network in fixed intervals
        self.Q_target = copy.deepcopy(self.Q)

        if self.config['use_existing_weights']:
            try:
                self.load_weights(os.path.join(self.config['weight_path'], "weights"))
                print("Using existing weights from " + self.config['weight_path'])
            except:
                print("Could not use existing weights")
        else:
            print("Not using existing weights")


    # keep this as it is
    def update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())


    # keep this as it is
    def act(self, observation, eps=None):
        if eps is None:
            eps = self.e
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self.action_space.sample()
            # added this line since the sample operation just returns an array of size 8 (actions)
            action = np.argmax(action, axis=-1)
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self):

        # Keep this as it is
        if self.PER:
            data, weights, indices = self.buffer.sample(batch=self.config['batch_size'])
        else:
            data = self.buffer.sample(batch=self.config['batch_size'])
        s = np.stack(data[:,0]) # s_t
        a = np.stack(data[:,1]) # a_t
        reward = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
        s_prime = np.stack(data[:,3]) # s_t+1
        done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)

        if self.config["use_target_net"]:
            v_prime = self.Q_target.maxQ(s_prime)[:, None]
        else:
            v_prime = self.Q.maxQ(s_prime)[:, None]
            # target
        gamma=self.config['discount']
        td_target = reward + gamma * (1.0-done) * v_prime

        if not self.PER:
            weights = np.ones(td_target.shape)

            # optimize the lsq objective
        fit_loss, pred = self.Q.fit(observations=s, actions=a, targets=td_target, weights=weights)

        # Update the priorities of the transitions inside the buffer
        if self.PER:
            self.buffer.update_priority(idx=indices[:,None], priority=abs(td_target-pred.detach().numpy()))

        return fit_loss

    def save_weights(self, filepath):
        T.save(self.Q.state_dict(), filepath)

    def load_weights(self, filepath):
        self.Q.eval()
        self.Q.load_state_dict(T.load(filepath))

    def print_transitions(self):
        print(self.buffer.get_all_transitions())
