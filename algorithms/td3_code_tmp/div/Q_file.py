import torch
import importlib.util

if importlib.util.find_spec("intel_extension_for_pytorch") is not None:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
else:
    ipex_available = False

# ipex_available = False ### FOR TESTING PURPOSES
if ipex_available and torch.xpu.is_available():
    device = torch.device("xpu")
    print("using xpu device")
else:
    device = torch.device("cpu")
    ipex_available = False
    print("using cpu device")

from torch import nn

from F_file import F

switch = False

class Q_net(F):

    def __init__(self):
        super(Q_net, self).__init__(use_ipex=ipex_available, comp_device=device)
        self = self.to(device)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3, eps=0.000001)

        self, self.opt = ipex.optimize(self, optimizer=self.opt, dtype=torch.bfloat16)

        self.loss = nn.SmoothL1Loss()
        self.loss = self.loss.to(device)

    def fit(self, o, a, t):
        self.train()
        self.opt.zero_grad()
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    q = self.Q_calc(o, a)
            loss_val = self.loss(q, t)

        loss_val.backward()
        self.opt.step()
        return loss_val.item()

    def Q_calc(self, o, a):
        if switch:
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                return self.forward(torch.hstack((o, a)))
        else:
            return self.forward(torch.hstack((o, a)))