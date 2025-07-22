import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque
np.random.seed(1)
torch.manual_seed(1)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

