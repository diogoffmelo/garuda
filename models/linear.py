import numpy as np
import tensorflow as tf
import h5py

from util.layers import full_layer, char_accuracy, xentropy_loss, word_accuracy
from util.batch import NumpyBatchGenerator, SampleMode
from util.report import report


db_path = '../generate/datatest.hdf5'
nepochs = 1

with h5py.File(db_path, 'r') as datadb:
    idxs = datadb['train']
    Xdata = datadb['X'].value[idxs]
    Ydata = datadb['Y'].value[idxs]
    idxs = datadb['test']
    Xdatat = datadb['X'].value[idxs]
    Ydatat = datadb['Y'].value[idxs]


Xsize, Xwidth, Xlength, Xdepth = list(Xdata.shape)
Xresized = Xwidth * Xlength * Xdepth
Ysize, Ypos, Yclasses = list(Ydata.shape)
batchgen = NumpyBatchGenerator(Xdata, Ydata, batch_size=100)
batchgenloss = NumpyBatchGenerator(Xdata, Ydata, SampleMode.SEQUENTIAL, batch_size=100)

Vsize, _, _, _ = Xdatat.shape
batchgenv = NumpyBatchGenerator(Xdatat, Ydatat, SampleMode.SEQUENTIAL, batch_size=Vsize)

Ps = list(range(Ypos))

Xtrue = tf.placeholder(tf.float32, [None, Xresized])
Ytrue = tf.placeholder(tf.float32, [None, Ypos, Yclasses])

def selector(p):
    _y = tf.reshape(Ytrue[:, p, :], [-1, Yclasses]) 
    _lin = full_layer(Xtrue, [Xresized, Yclasses], 'layer{}'.format(p), None)
    _ylogits = _lin['aout']
    _ypreds = tf.nn.softmax(_ylogits)
    _acc = char_accuracy(_y, _ypreds, 'acc{}'.format(p))
    _xent = xentropy_loss(_ylogits, _y, 'xent{}'.format(p))
    _t = tf.train.AdamOptimizer(0.001).minimize(_xent['xentropy'])
    return (_y, _lin, _ylogits, _ypreds, _acc, _xent, _t), _acc['accuracy'], _xent['loss']

model = list(map(selector, Ps))
train = [x[0][6] for x in model]
loss = [x[2] for x in model]
accuracy = [x[1] for x in model]
preds = [x[0][4]['correct_pred'] for x in model]
_wacc = word_accuracy(preds, accuracy, 'wacc')

mpacc = _wacc['pacc']
mwacc = _wacc['wacc']

def food(batch):
    _X, _Y = batch
    return {Ytrue: _Y, Xtrue: _X.reshape([-1, Xresized])}


class Model(object):
    def __init__(self):
        self.metrics = {
            'cacc': accuracy,
            'loss': loss,
            'wacc': mwacc, 
            'pacc': mpacc,
        }

    def food(self, batch):
        _X, _Y = batch
        return {Ytrue: _Y, Xtrue: _X.reshape([-1, Xresized])}


linear = Model()

def epoch_hook(epoch):
    if epoch%3 == 0:
        print('Train @t={}'.format(epoch))
        print(report(linear, batchgenloss, sess))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for batch in batchgen.gen_batch_epochs(nepochs, epoch_hook):
        sess.run(train, feed_dict=food(batch))

    print('Vadation:')
    print(report(linear, batchgenv, sess))
