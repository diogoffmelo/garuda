from enum import Enum
import h5py

from .batch import SampleMode, NumpyBatchGenerator


class BatchMode(Enum):
    NUMPY = 0


def load_from_HDF5_to_memory(dbargs):
    with h5py.File(dbargs['path'], 'r') as datadb:
        X = datadb['X'].value
        Y = datadb['Y'].value
        
        idxs = datadb['train']
        Xtrain = X[idxs]
        Ytrain = Y[idxs]

        idxs = datadb['test']
        Xtest = X[idxs]
        Ytest = Y[idxs]

    return Xtrain, Ytrain, Xtest, Ytest


def load_to_batches(dbargs,
                    batchargs,
                    mode=BatchMode.NUMPY,
                    dbfunc=load_from_HDF5_to_memory):

    Xtrain, Ytrain, Xtest, Ytest = dbfunc(dbargs)

    train_bsize = batchargs.get('train_bsize', 100)
    report_bsize = batchargs.get('report_bsize', 100)
    val_bsize = batchargs.get('val_bsize', 0)
    if not val_bsize:
        val_bsize, _, _, _ = Xtest.shape 
    
    train_gen = NumpyBatchGenerator(Xtrain, 
                                    Ytrain,
                                    SampleMode.SEQUENTIAL, 
                                    batch_size=train_bsize)
    
    report_gen = NumpyBatchGenerator(Xtrain,
                                     Ytrain,
                                     SampleMode.SEQUENTIAL,
                                     batch_size=report_bsize)
    
    val_gen = NumpyBatchGenerator(Xtest,
                                  Ytest,
                                  SampleMode.SEQUENTIAL,
                                  batch_size=val_bsize)
    
    return train_gen, report_gen, val_gen
