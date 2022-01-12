import random
import os, sys, pickle
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def pop(self, idx):
        if idx>self.__len__():
            print("err:memory index")
            sys.exit()
        return self.memory[idx]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)
    
    def save(self, save_dir):
        f = open(os.path.join(save_dir, 'buffer'), 'wb')
        pickle.dump(self.memory, f)

    def load(self, load_dir):
        self.memory=[]

        f = open(os.path.join(load_dir, 'buffer'),"rb")
        self.memory = pickle.load(f)

    def __len__(self):
        return len(self.memory)
