import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

## For source see https://www.youtube.com/watch?v=H9uCYnG3LlE
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, input_dims, device):
        super(DuelingDeepQNetwork, self).__init__()

        if device.type == 'cuda':
            self.cuda()
        ## TODO try different sizes of the network,
        ##  e.g. add layers or change number of neurons (maybe add another layer before both A & V)
        self.fc1 = nn.Linear(input_dims, 256)

        # TODO maybe implement this as a list later to add more layers
        self.A1 = nn.Linear(256,256)
        self.V1 = nn.Linear(256,256)
        self.A2 = nn.Linear(256,256)
        self.V2 = nn.Linear(256,256)

        self.A_stream = nn.Linear(256, 8)
        self.V_stream = nn.Linear(256, 1)


    def forward(self, x):
        if self.device.type == 'cuda' and x.device.type != 'cuda':
            x = x.to(self.device)

        x = nn.functional.relu(self.fc1(x))

        x_A = nn.functional.relu(self.A1(x))
        x_V = nn.functional.relu(self.V1(x))
        x_A = nn.functional.relu(self.A2(x_A))
        x_V = nn.functional.relu(self.V2(x_V))

        A = self.A_stream(x_A)
        V = self.V_stream(x_V)

        Q = T.add(V, (A - A.mean(dim=-1, keepdim=True)))
        return Q

    ## TODO Maybe adapt this a bit (source)
    def predict(self, x):
        with T.no_grad():
            return self.forward(T.from_numpy(x.astype(np.float32))).cpu().numpy()

    def save_checkpoint(self):
        print('-> Saving Checkpoint')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('-> Loading Checkpoint')
        self.load_state_dict(T.load(self.checkpoint_file))


## Took this class 1:1 from the exercises and made minor changes/enhancements
class QFunction(DuelingDeepQNetwork):
    def __init__(self, observation_dim, action_dim,
                 lr, device):
        super().__init__(input_dims=observation_dim,
                         device=device)

        if device.type == 'cuda':
            self.cuda()

        self.loss = nn.SmoothL1Loss() # MSELoss()
        self.lr = lr
        self.optimizer=optim.Adam(self.parameters(),
                                  lr=self.lr)
        self.device = device

    def fit(self, observations, actions, targets, weights):
        # Forward pass
        prediction = self.Q_value(observations=observations, actions=actions)
        # Compute Loss and weight it
        weighted_loss = self.loss(prediction, T.from_numpy(targets).to(self.device).float()) * (T.from_numpy(weights).to(self.device).float())
        self.optimizer.zero_grad()
        # Backward pass
        mean_wl = weighted_loss.mean()
        mean_wl.backward()
        self.optimizer.step()
        return mean_wl.item(), prediction


    def Q_value(self, observations, actions):
        return T.gather(self.forward(T.from_numpy(observations).to(self.device).float()), 1, T.from_numpy(actions[:,None]).to(self.device).long())

    def maxQ(self, observations):
        # compute the maximal Q-value
        return np.max(self.predict(observations), axis=-1)

    def greedyAction(self, observations):
        # this computes the greedy action
        start_predict = time.time()
        i = np.argmax(self.predict(observations), axis=-1)
        #print("predict time: " + str(time.time() - start_predict))
        return i









