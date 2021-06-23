import torch
import torch.utils.data
import numpy as np
import random, math

class RandomBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, examples_num, batch_size, drop_last):
        self.examples_num = examples_num
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            self._len = self.examples_num // self.batch_size
        else:
            self._len = (self.examples_num + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for _ in range(self._len):
            # NOTE: for small batches (i.e. <= 3000), random.sample is way
            # faster than np.random.choice for sampling without replacement.
            batch = random.sample(range(self.examples_num), self.batch_size)
            yield batch

    def __len__(self):
        return self._len