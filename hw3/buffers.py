import numpy as np
import random
from collections import deque


class Buffer(object):

    def __init__(self, size, sample_size):

        self.size = size
        self.sample_size = sample_size
        self.buffer = deque()

    def add(self, state, action, reward, next_state):
        self.buffer.append( (state, action, reward, next_state) )

        if len(self.buffer) > self.size:
            self.buffer.popleft()

    def sample(self):
        if len(self.buffer) < self.sample_size:
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, self.sample_size)

        state = np.reshape(np.array([arr[0] for arr in samples]), [len(samples), -1])
        action = np.array([arr[1] for arr in samples])
        reward = np.array([arr[2] for arr in samples])
        next_state = np.reshape(np.array([arr[3] for arr in samples]), [len(samples), -1])

        return state, action, reward, next_state
