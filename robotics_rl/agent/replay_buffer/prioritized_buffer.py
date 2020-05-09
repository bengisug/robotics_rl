import numpy as np

from robotics_rl.agent.replay_buffer.sum_tree import SumTree


class PrioritizedBuffer(object):

    def __init__(self, capacity, tuple_type, min_prioirty=0.001, max_priority=2):
        self.queue = []
        self.capacity = capacity
        self.sumtree = SumTree(capacity)
        self.min_prioirty = min_prioirty
        self.max_priority = max_priority
        self.tuple_type = tuple_type

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return "PrioritizedBuffer"

    @property
    def size(self):
        """Return the current size of the buffer"""
        return len(self.queue)

    def _clip_p(self, p):
        """ Return clipped priority """
        return int(min(max(p, self.min_prioirty), self.max_priority) * (10**6))

    @property
    def cycle(self):
        """ Return the current writing head """
        return self.sumtree.cycle

    def reset(self):
        self.queue = []
        self.sumtree.tree = np.zeros(shape=self.sumtree._tree_size * 2 - 1, dtype="float64")
        self.sumtree.current_size = 0
        self.sumtree.cycle = 0

    def push(self, *args):
        """ Push the transition with priority """
        priority = args[-1]
        transition = args[:-1]
        if (self.size < self.capacity):
            self.queue.append(self.tuple_type(*transition))
        else:
            self.queue[self.cycle] = self.tuple_type(*transition)

        self.sumtree.push(self._clip_p(priority))

    def sample(self, batch_size):
        """ Return namedtuple of transition that
        is sampled with probability proportional to
        the priority values. """

        if batch_size > len(self.queue):
            return None

        max_value = self.sumtree.tree[0]
        slices = np.linspace(0, max_value, num=batch_size + 1)
        values = np.random.uniform(slices[:-1], slices[1:], size=batch_size)
        indexes = [self.sumtree.get(value) for value in values]

        batch = self.tuple_type(*zip(*(self.queue[idx] for idx in indexes)))

        self.indexes = indexes

        return batch

    def update_priority(self, values):
        """ Update the priority value of the transition in
        the given index
        """
        for idx, value in zip(self.indexes, values):
            self.sumtree.update(idx, self._clip_p(value))