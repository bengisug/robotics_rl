""" Uniform Buffer
"""
from random import sample as randsample


class UniformBuffer(object):

    def __init__(self, size, tuple_type):
        self.queue = []
        self.cycle = 0
        self.size = size
        self.tuple_type = tuple_type

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return "UniformBuffer"

    def reset(self):
        self.queue = []
        self.cycle = 0

    def push(self, *transition):
        if self.size != len(self.queue):
            self.queue.append(self.tuple_type(*transition))
        else:
            self.queue[self.cycle] = self.tuple_type(*transition)
            self.cycle = (self.cycle + 1) % self.size

    def sample(self, batch_size):
        if batch_size > len(self.queue):
            return None
        batch = randsample(self.queue, batch_size)
        return self.tuple_type(*zip(*batch))