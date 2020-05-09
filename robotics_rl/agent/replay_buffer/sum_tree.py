""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SumTree():
    """ Binary heap with the property: parent node is the sum of
    two child nodes. Tree has a maximum size and whenever
    it reaches that, the oldest element will be overwritten
    (queue behaviour). All of the methods run in O(log(n)).

    Arguments
        - maxsize: Capacity of the SumTree

    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.current_size = 0
        self.cycle = 0
        self._tree_size = 2**int(np.ceil(np.log2(maxsize)))
        self.tree = np.zeros(shape=self._tree_size*2-1, dtype="float64")

    def push(self, priority):
        """ Add an element to the tree and with the given priority.
         If the tree is full, overwrite the oldest element.

        Arguments
            - priority: Corresponding priority value
        """
        self.update(self.cycle, priority)
        self.cycle = (self.cycle + 1) % self.maxsize
        self.current_size = min(self.current_size + 1, self.maxsize)

    def get(self, priority):
        """ Return the node with the given priority value.
        Prioirty can be at max equal to the value of the root
        in the tree.

        Arguments
            - priority: Value whose corresponding index
                will be returned.
        """
        assert priority <= self.tree[0], """Given prioirty value is 
            larger than the max value"""
        node = 0
        while (node*2 + 1 < self.tree.size):
            left = node*2+1
            right = left+1
            if (priority <= self.tree[left]):
                node = left
            else:
                priority -= self.tree[left]
                node = right
        node -= self._tree_size - 1
        assert node < self.current_size, ("Unavailable index: {}, size: {}!"
                                          .format(node, self.current_size))
        return node

    def update(self, idx, value):
        """ Update the tree for the given idx with the
        given value. Values are updated via increasing
        the priorities of all the parents of the given
        idx by the difference between the value and
        current priority of that idx.

        Arguments
            - idx: Index of the data(not the tree).
            Corresponding index of the tree can be
            calculated via; idx + tree_size/2 - 1
            - value: Value for the node at pointed by
            the idx
        """
        parent = idx + self._tree_size - 1
        delta = value - self.tree[parent]
        self.tree[parent] += delta
        while(parent != 0):
            parent = (parent - 1)//2
            self.tree[parent] += delta
