import torch
import numpy as np

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.tanh, output_activation=None, use_batch_norm=True):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o, bias=(not use_batch_norm)) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        if use_batch_norm:
            self.batch_norms = torch.nn.ModuleList([ torch.nn.LayerNorm(o) for o in layer_sizes[1:]])
        else:
            self.batch_norms = torch.nn.ModuleList([ torch.nn.Identity() for o in layer_sizes[1:]])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer, batch_layer, activation_fun in zip(self.layers, self.batch_norms, self.activations):
            x = layer(x)
            x = batch_layer(x)
            x = activation_fun(x)
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        x = to_torch(x)
        with torch.no_grad():
            return self.forward(x).numpy()
