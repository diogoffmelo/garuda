import numpy as np
import tensorflow as tf
import h5py

from util.layers import full_layer, char_accuracy, xentropy_loss, word_accuracy
from util.batch import NumpyBatchGenerator, SampleMode


db_path = '../generate/datatest.hdf5'
nepochs = 300

with h5py.File(db_path, 'r') as datadb:
    idxs = datadb['train']
    Xdata = datadb['X'].value[idxs]
    Ydata = datadb['Y'].value[idxs]

Xsize, Xwidth, Xlength, Xdepth = list(Xdata.shape)
Xresized = Xwidth * Xlength * Xdepth
Ysize, Ypos, Yclasses = list(Ydata.shape)
batchgen = NumpyBatchGenerator(Xdata, Ydata, batch_size=100)
batchgenloss = NumpyBatchGenerator(Xdata, Ydata, SampleMode.SEQUENTIAL, batch_size=100)

#Xtrue = tf.placeholder(tf.float32, [None, Xresized])
#Ytrue = tf.placeholder(tf.float32, [None, Yclasses])


# layer1 = full_layer(Xtrue, [Xresized, Yclasses], 'layer1', None)
# Ylogits = layer1['aout']

# Ypreds = tf.nn.softmax(Ylogits)
# def food(batch):
#     _X, _Y = batch
#     return {
#         Xtrue: _X.reshape([-1, Xresized]),
#         Ytrue: _Y[:, 0, :].reshape([-1, Yclasses])}


# correct_preds = tf.equal(tf.argmax(Ypreds, 1), tf.argmax(Ytrue, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Ytrue)
# loss = tf.reduce_mean(xentropy) * 1000
# train = tf.train.AdamOptimizer(0.001).minimize(xentropy)


# Xtrue = tf.placeholder(tf.float32, [None, Xresized])
# Ytrues = []
# layers = []
# Ylogits = []
# correct_preds = []
# Ypreds = []
# accuracy = []
# xentropy = []
# loss = []
# train = []
# for p in range(Ypos):
#     Ytrues.append(tf.placeholder(tf.float32, [None, Yclasses]))
#     layers.append(full_layer(Xtrue, [Xresized, Yclasses], 'layer{}'.format(p), None))
#     Ylogits.append(layers[p]['aout'])
#     Ypreds.append(tf.nn.softmax(Ylogits[p]))
#     correct_preds.append(tf.equal(tf.argmax(Ypreds[p], 1), tf.argmax(Ytrues[p], 1)))
#     accuracy.append(tf.reduce_mean(tf.cast(correct_preds[p], tf.float32)))
#     xentropy.append(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits[p], labels=Ytrues[p]))
#     loss.append(tf.reduce_mean(xentropy[p]) * 1000)
#     train.append(tf.train.AdamOptimizer(0.001).minimize(xentropy[p]))


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
_wacc = word_accuracy(preds, 'wacc')
wacc = _wacc['accuracy'] 

def food(batch):
    _X, _Y = batch
    return {Ytrue: _Y, Xtrue: _X.reshape([-1, Xresized])}


# Ytrues = []
# layers = []
# Ylogits = []
# correct_preds = []
# Ypreds = []
# accuracy = []
# xentropy = []
# loss = []
# train = []
# for p in range(Ypos):
#     Ytrues.append(tf.placeholder(tf.float32, [None, Yclasses]))
#     layers.append(full_layer(Xtrue, [Xresized, Yclasses], 'layer{}'.format(p), None))
#     Ylogits.append(layers[p]['aout'])
#     Ypreds.append(tf.nn.softmax(Ylogits[p]))
#     correct_preds.append(tf.equal(tf.argmax(Ypreds[p], 1), tf.argmax(Ytrues[p], 1)))
#     accuracy.append(tf.reduce_mean(tf.cast(correct_preds[p], tf.float32)))
#     xentropy.append(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits[p], labels=Ytrues[p]))
#     loss.append(tf.reduce_mean(xentropy[p]) * 1000)
#     train.append(tf.train.AdamOptimizer(0.001).minimize(xentropy[p]))
# def food(batch):
#     _X, _Y = batch
#     food = {Ytrues[p]: _Y[:, p, :].reshape([-1, Yclasses])  for p in range(Ypos)}
#     food[Xtrue] = _X.reshape([-1, Xresized])
#     return food


def epoch_hook(epoch):
    if epoch%3 == 0:
        _a, _l, N = [0 for _ in range(Ypos)], [0 for _ in range(Ypos)], 0
        _w = 0.0
        for batch in batchgenloss.gen_batches():
            a, l, w = sess.run([accuracy, loss, wacc], feed_dict=food(batch))
            _a = [_a[p] + a[p] for p in range(Ypos)]
            _l = [_l[p] + l[p] for p in range(Ypos)]
            _w += w
            N = N + 1

        print('@t={}'.format(epoch))
        ___a = ','.join(['{:.2f}'.format(_a[p]/N) for p in range(Ypos)])
        print('acc=[{}]'.format(___a))
        print('wcc=[{}]'.format(w/N))
        # print("Train acc={} wacc={} xent={}".format(
        #         [_a[p]/N for p in range(Ypos)],
        #         w/N,
        #         [_l[p]/N for p in range(Ypos)]))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for batch in batchgen.gen_batch_epochs(nepochs, epoch_hook):
        sess.run(train, feed_dict=food(batch))


    # with h5py.File(db_path, 'r') as datadb:
    #     idxs = datadb['test']
    #     Xdatav = datadb['X'].value[idxs]
    #     Ydatav = datadb['Y'].value[idxs]


    # batchgenlossv = NumpyBatchGenerator(Xdatav, Ydatav, SampleMode.SEQUENTIAL, 100)

    # _a, _l, N = [0 for _ in range(Ypos)], [0 for _ in range(Ypos)], 0
    # for batch in batchgenlossv.gen_batches():
    #     a, l = sess.run([accuracy, loss], feed_dict=food(batch))
    #     _a = [_a[p] + a[p] for p in range(Ypos)]
    #     _l = [_l[p] + l[p] for p in range(Ypos)]
    #     N = N + 1

    # print("Test accuracy={} entropy={}".format(
    #         [_a[p]/N for p in range(Ypos)],
    #         [_l[p]/N for p in range(Ypos)]))
