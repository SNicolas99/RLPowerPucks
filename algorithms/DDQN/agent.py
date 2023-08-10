import copy
from functools import reduce

import numpy as np
import torch as T
from algorithms.DDQN.memory import Memory, PER_Memory
from algorithms.DDQN.q_dueling_network import QFunction
import os
import time


## IMPORANT NOTICE: This class is built primarily on the exericise code of the reinforcement learning course at the University of Tübingen
class DQNAgent():

    def __init__(self, observation_space, action_space, config=False, tournament_weights=False, tournament_call=False):

        self.observation_dim = observation_space.shape[0]
        self.action_space = action_space
        self.config = config
        print("tournament call: " + str(tournament_call))

        if tournament_call is True:
            self.config ={
                "epsilon": 0,
                "PER": True,
                "learning_rate": 0.0001,
                "device": None,
                "use_existing_weights": True,
                "weight_path": tournament_weights,
                "batch_size": 32,
                "discount": 0.98,
                "dueling_architecture": True,
                "buffer_size":2000000,
                "beta": 0.4,
                "alpha": 0.6
            }


        self.e = self.config['epsilon']
        self.PER = self.config['PER']

        if self.PER is True:
            self.buffer = PER_Memory(max_size=self.config["buffer_size"], beta=self.config['beta'], alpha=self.config['alpha'])
        else:
            self.buffer = Memory(max_size=self.config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self.observation_dim,
                           action_dim=self.action_space,
                           lr = self.config['learning_rate'],
                           device=self.config['device'],
                           config=self.config)

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
        if self.PER is True:
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

        if self.PER is True:
            fit_loss, pred = self.Q.PER_fit(observations=s, actions=a, targets=td_target, weights=weights)
            self.buffer.update_priority(idx=indices[:,None], priority=abs(td_target-pred.detach().numpy()))
        else:
            fit_loss = self.Q.fit(observations=s, actions=a, targets=td_target)

        return fit_loss

    def save_weights(self, filepath):
        T.save(self.Q.state_dict(), filepath)

    def load_weights(self, filepath):
        self.Q.eval()
        self.Q.load_state_dict(T.load(filepath))

    def print_transitions(self):
        print(self.buffer.get_all_transitions())
