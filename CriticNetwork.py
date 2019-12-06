import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_NEW = 32
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Linear(state_size, HIDDEN_NEW)
        self.w1 = nn.Linear(HIDDEN_NEW, HIDDEN1_UNITS)
        self.a1 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h3 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.V = nn.Linear(HIDDEN2_UNITS, action_size)

    def forward(self, s, a):
        s = F.relu(self.fc(s))
        w1 = F.relu(self.w1(s))
        a1 = self.a1(a)
        h1 = self.h1(w1)
        h2 = h1 + a1
        h3 = F.relu(self.h3(h2))
        out = self.V(h3)
        return out
