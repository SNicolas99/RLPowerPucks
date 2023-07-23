import torch
import torch.nn as nn


# define ffw network
class FeedforwardNetwork(nn.Module):
    def __init__(
        self, input_size, output_size, act=torch.tanh, act_out=torch.tanh, h=256
    ):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, output_size)
        self.act = act
        self.act_out = act_out

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.act_out(self.fc3(x))

    def copy(self):
        new_network = FeedforwardNetwork(
            input_size=self.fc1.in_features,
            output_size=self.fc3.out_features,
            act=self.act,
            act_out=self.act_out,
            h=self.fc1.out_features,
        )
        new_network.load_state_dict(self.state_dict())
        return new_network
