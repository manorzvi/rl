import torch
from collections import deque


class StackedStates:

    def __init__(self, maxlen: int = 4):
        self.maxlen = maxlen

    def reset(self, state: torch.Tensor):
        self.stack = deque(maxlen=self.maxlen)
        for i in range(self.maxlen):
            self.stack.append(state)

    def push(self, state: torch.Tensor):
        self.stack.append(state)

    def __call__(self):
        return torch.stack(tuple(self.stack), dim=0)
