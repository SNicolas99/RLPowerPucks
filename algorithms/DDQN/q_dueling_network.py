import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


## For source see https://www.youtube.com/watch?v=H9uCYnG3LlE
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, input_dims, device, config):
        super(DuelingDeepQNetwork, self).__init__()
        self.config = config

        if self.config['dueling_architecture'] is True:
            self.fc1 = nn.Linear(input_dims, 256)

            self.A1 = nn.Linear(256, 256)
            self.V1 = nn.Linear(256, 256)
            self.A2 = nn.Linear(256, 256)
            self.V2 = nn.Linear(256, 256)

            self.A_stream = nn.Linear(256, 8)
            self.V_stream = nn.Linear(256, 1)
        else:
            self.fc1 = nn.Linear(input_dims, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.Q = nn.Linear(256, 8)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))

        if self.config['dueling_architecture'] is True:
            x_A = nn.functional.relu(self.A1(x))
            x_V = nn.functional.relu(self.V1(x))
            x_A = nn.functional.relu(self.A2(x_A))
            x_V = nn.functional.relu(self.V2(x_V))

            A = self.A_stream(x_A)
            V = self.V_stream(x_V)

            return T.add(V, (A - A.mean(dim=-1, keepdim=True)))
        else:
            x_fc2 = nn.functional.relu(self.fc2(x))
            x_fc3 = nn.functional.relu(self.fc3(x_fc2))
            x_Q = nn.functional.relu(self.Q(x_fc3))

            return x_Q

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
                 lr, device, config):
        super().__init__(input_dims=observation_dim,
                         device=device,
                         config=config)

        self.loss = nn.SmoothL1Loss()  # MSELoss()
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.lr)
        self.device = device
        self.config = config

    def PER_fit(self, observations, actions, targets, weights):
        # Forward pass
        prediction = self.Q_value(observations=observations, actions=actions)
        # Compute Loss and weight it
        weighted_loss = self.loss(prediction, T.from_numpy(targets).to(self.device).float()) * (
            T.from_numpy(weights).to(self.device).float())
        self.optimizer.zero_grad()
        # Backward pass
        mean_wl = weighted_loss.mean()
        mean_wl.backward()
        self.optimizer.step()
        return mean_wl.item(), prediction

    def fit(self, observations, actions, targets):
        self.train()
        self.optimizer.zero_grad()
        # Forward pass
        prediction = self.Q_value(observations=observations, actions=actions)
        # Compute Loss and weight it
        weighted_loss = self.loss(prediction, T.from_numpy(targets).to(self.device).float())
        # Backward pass
        mean_wl = weighted_loss.mean()
        mean_wl.backward()
        self.optimizer.step()
        return mean_wl.item()

    def Q_value(self, observations, actions):
        return T.gather(self.forward(T.from_numpy(observations).to(self.device).float()), 1,
                        T.from_numpy(actions[:, None]).to(self.device).long())

    def maxQ(self, observations):
        # compute the maximal Q-value
        return np.max(self.predict(observations), axis=-1)

    def greedyAction(self, observations):
        # this computes the greedy action
        return np.argmax(self.predict(observations), axis=-1)
