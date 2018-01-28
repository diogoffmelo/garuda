import numpy as np
import tensorflow as tf
import h5py

from util.batch import NumpyBatchGenerator, SampleMode
from util.report import report
from models.linear import LinearW
from models.mlp import Mlp

db_path = '../generate/datatest.hdf5'
nepochs = 30

with h5py.File(db_path, 'r') as datadb:
    idxs = datadb['train']
    Xdata = datadb['X'].value[idxs]
    Ydata = datadb['Y'].value[idxs]
    idxs = datadb['test']
    Xdatat = datadb['X'].value[idxs]
    Ydatat = datadb['Y'].value[idxs]

_, Xwidth, Xlength, Xdepth = Xdata.shape
Xresized = Xwidth * Xlength * Xdepth
_, Ypos, Yclasses = list(Ydata.shape)
batchgen = NumpyBatchGenerator(Xdata, Ydata, batch_size=100)
batchgenloss = NumpyBatchGenerator(Xdata, Ydata, SampleMode.SEQUENTIAL, batch_size=100)

Vsize, _, _, _ = Xdatat.shape
batchgenv = NumpyBatchGenerator(Xdatat, Ydatat, SampleMode.SEQUENTIAL, batch_size=Vsize)

graph = tf.Graph()
with graph.as_default():
    Xtrue = tf.placeholder(tf.float32, [None, Xresized])
    Ytrue = tf.placeholder(tf.float32, [None, Ypos, Yclasses])

#model = LinearW(Xtrue, Ytrue, graph, 'linear')
model = Mlp(Xtrue, Ytrue, 1000, graph, 'mlp')

with graph.as_default():
    train = [tf.train.AdamOptimizer(0.001).minimize(x) for x in model.xent]


with tf.Session(graph=model.g) as sess:
    def epoch_hook(epoch):
        if epoch%3 == 0:
            print('Train @t={}'.format(epoch))
            print(report(model, batchgenloss, sess))

    init = tf.global_variables_initializer()
    sess.run(init)

    for batch in batchgen.gen_batch_epochs(nepochs, epoch_hook):
        sess.run(train, feed_dict=model.food(batch))

    print('Vadation:')
    print(report(model, batchgenv, sess))
