import torch
from torch import nn

import numpy as np

def to_torch_device(x, device=torch.device("cpu")):
    if not isinstance(x, torch.Tensor):
        if device == torch.device("cpu"):
            return torch.from_numpy(x.astype(np.float32))
        else:
            return torch.from_numpy(x.astype(np.float16)).to(device, dtype=torch.bfloat16)
    else:
        if device == torch.device("cpu"):
            return x.to(dtype=torch.float32)
        else:
            return x.to(device, dtype=torch.bfloat16)

# class F(nn.Module):
#     def __init__(self, activation_fun=torch.sigmoid, output_activation=None, use_ipex=False, comp_device=torch.device("cpu")):
#         super(F, self).__init__()
#         self.input_size = 4 # input_size
#         self.hidden_sizes  = [100, 100] # hidden_sizes
#         self.output_size  = 1  # output_size
#         self.output_activation = output_activation
#         layer_sizes = [self.input_size] + self.hidden_sizes
#         self.layers = nn.Sequential( *[torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
#         self.activations = [ activation_fun for l in  self.layers ]
#         self.readout = nn.Linear(self.hidden_sizes[-1], self.output_size)
#         self.use_ipex = use_ipex
#         self.comp_device = comp_device
#         self.activation_fun = activation_fun

#         # self.to_torch = lambda x: to_torch_device(x, device=self.comp_device)

#     def forward(self, x):
#         if self.use_ipex:
#             with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
#                 for layer,activation_fun in zip(self.layers, self.activations):
#                     x = self.activation_fun(layer(x))
#                 if self.output_activation is not None:
#                     return self.output_activation(self.readout(x))
#                 else:
#                     return self.readout(x)
#         else:
#             for layer,activation_fun in zip(self.layers, self.activations):
#                 x = activation_fun(layer(x))
#             if self.output_activation is not None:
#                 return self.output_activation(self.readout(x))
#             else:
#                 return self.readout(x)

# import torch
# from torch import nn

class F(nn.Module):
    def __init__(self, use_ipex=False, comp_device=torch.device("cpu")):
        super(F, self).__init__()
        self.fc1 = nn.Linear(4, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            x = self.fc1(x)
            x = torch.tanh(x)
            x = self.fc2(x)
        return x