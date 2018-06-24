import pickle

import pandas as pd
import tensorfow as tf
import numpy as np

import time

from util.batch import NumpyBatchGenerator, SampleMode
from util.load import load_to_batches


if TEST:
    nepochs = 3
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }
else:
    nepochs = 50
    dbargs = {
        'path': '../generate/data.hdf5',
    }
    batchargs = {
        'train_bsize': 30,
        'report_bsize': 30,
        'val_bsize': 30,
    }

train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)
max_learning_rate = 0.01
min_learning_rate = 0.0001


