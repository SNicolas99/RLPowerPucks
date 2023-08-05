import torch
import intel_extension_for_pytorch as ipex

device = torch.device("xpu:0")

from torch import nn

class F(nn.Module):
    def __init__(self, use_ipex=False, comp_device=torch.device("cpu")):
        super(F, self).__init__()
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x, act):
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            x = self.fc1(x)
            x = act(x)
            x = self.fc2(x)
        return x

net = F()
net = net.to(device)

x = torch.randn((1,1), device=device)
target = torch.randn((1,1), device=device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# optimizing makes no difference
# net, optimizer = ipex.optimize(net, optimizer=optimizer, dtype=torch.bfloat16)

loss = nn.MSELoss()
# loss = loss.to(device) # makes no difference

############################################################

with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    y = torch.tanh(net.forward(x, torch.relu))
    loss_val = loss(y, target)
loss_val.backward()

print("relu result:", loss_val.item())

############################################################

mytanh = lambda x: 2 * torch.sigmoid(2 * x) - 1
with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    y = torch.tanh(net.forward(x, mytanh))
    loss_val = loss(y, target)
loss_val.backward()

print("custom tanh result:", loss_val.item())

############################################################

x_1d = torch.randn(1, device=device)
target_1d = torch.randn(1, device=device)
with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    y = torch.tanh(net.forward(x_1d, torch.tanh))
    loss_val = loss(y, target_1d)
loss_val.backward()

print("tanh 1d result:", loss_val.item())

############################################################

with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    y = torch.tanh(net.forward(x, torch.tanh))
    loss_val = loss(y, target)
loss_val.backward() # crashes here <------------------------

print("tanh result:", loss_val.item())
