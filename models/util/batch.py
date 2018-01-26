from enum import Enum
import numpy as np

class SampleMode(Enum):
    SEQUENTIAL = 0
    RANDOM = 1


class BatchGeneratorMixin(object):
    def __init__(self, mode, set_size, batch_size=128):
        self.mode = mode
        self.set_size = set_size
        self.batch_size = batch_size
        self.idxs = np.arange(set_size)
        self.max_seek = int(set_size/batch_size)

    def gen_idxs(self):
        seek = 0
        while seek < self.max_seek:
            if self.mode == SampleMode.SEQUENTIAL:
                low = seek * self.batch_size
                high = low + self.batch_size
                yield self.idxs[low:high]

            elif self.mode == SampleMode.RANDOM:
                yield np.random.choice(self.idxs, self.batch_size)

            seek += 1


class BatchGeneratorEpochsMixin(object):
    def gen_batch_epochs(self, nepochs, ith_iteration_hook=None):    
        for i in range(nepochs):
            for batch in self.gen_batches():
                yield batch

            if ith_iteration_hook:
                ith_iteration_hook(i)


class NumpyBatchGenerator(BatchGeneratorMixin, BatchGeneratorEpochsMixin):
    def __init__(self, X, Y, mode=SampleMode.RANDOM, batch_size=128):
        self.X = X
        self.Y = Y
        BatchGeneratorMixin.__init__(self, mode, X.shape[0], batch_size)

    def gen_batches(self):
        for idxs in self.gen_idxs():
            yield self.X[idxs], self.Y[idxs]
