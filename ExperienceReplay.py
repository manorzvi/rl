from collections import deque, namedtuple
from typing import List
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExperienceReplay:
    def __init__(self, capacity: int):

        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int, consecutive=False) -> List:
        buffer_size = len(self.buffer)
        if not consecutive:
            index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        else:
            index = np.random.choice(np.arange(buffer_size-batch_size+1), size=1, replace=False)
            index = np.arange(index, index + batch_size)

        return [self.buffer[i] for i in index]

    def __len__(self):
        return len(self.buffer)
