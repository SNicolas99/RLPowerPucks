# Memory class provided 
import numpy as np

class Memory():
    # class to store x/u trajectory
    def __init__(self, buffer_size=int(1e5)):
        self.xux = np.asarray([])
        self.size = 0

    def add_item(self, xux_new):
        if self.size is not 0:
            self.xux = np.concatenate([self.xux, np.asarray(xux_new).reshape(1,-1)])
        else:
            self.xux = np.asarray(xux_new).reshape(1,-1)
        self.size += 1

    def sample(self, batch=1):
        self.size = self.xux.shape[0]
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        return self.xux[self.inds,:]
