from enum import Enum
import numpy as np
import h5py


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
        self.xshape = X.shape[1:]
        self.yshape = Y.shape[1:]
        BatchGeneratorMixin.__init__(self, mode, X.shape[0], batch_size)

    def gen_batches(self):
        for idxs in self.gen_idxs():
            yield self.X[idxs], self.Y[idxs]


def load_from_HDF5_to_memory(path):
    with h5py.File(path, 'r') as datadb:
        X = datadb['X'].value
        Y = datadb['Y'].value
        
        idxs = datadb['train']
        Xtrain = X[idxs]
        Ytrain = Y[idxs]

        idxs = datadb['test']
        Xtest = X[idxs]
        Ytest = Y[idxs]

    return Xtrain, Ytrain, Xtest, Ytest


def load_to_batches(path, train_bsize=10, val_bsize=10):

    Xtrain, Ytrain, Xtest, Ytest = load_from_HDF5_to_memory(path)

    train_gen = NumpyBatchGenerator(Xtrain, 
                                    Ytrain,
                                    SampleMode.RANDOM, 
                                    batch_size=train_bsize)
    
    report_gen = NumpyBatchGenerator(Xtrain,
                                     Ytrain,
                                     SampleMode.SEQUENTIAL,
                                     batch_size=val_bsize)
    
    val_gen = NumpyBatchGenerator(Xtest,
                                  Ytest,
                                  SampleMode.SEQUENTIAL,
                                  batch_size=val_bsize)
    
    return train_gen, report_gen, val_gen
