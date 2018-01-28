import numpy as np
import tensorflow as tf

from util.batch import NumpyBatchGenerator, SampleMode
from util.report import report
from models import LinearW, Mlp
from util.load import load_to_batches


nepochs = 30
dbargs = {
    'path': '../generate/datatest.hdf5',
}
batchargs = {
    'train_bsize': 10,
    'report_bsize': 10,
    'val_bsize': 0,
}
train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)


Xwidth, Xlength, Xdepth = train_gen.xshape
Xresized = Xwidth * Xlength * Xdepth
Ypos, Yclasses = train_gen.yshape

graph = tf.Graph()
with graph.as_default():
    Xtrue = tf.placeholder(tf.float32, [None, Xresized])
    Ytrue = tf.placeholder(tf.float32, [None, Ypos, Yclasses])

model = LinearW(Xtrue, Ytrue, graph, 'linear')
#model = Mlp(Xtrue, Ytrue, 1000, graph, 'mlp')

with graph.as_default():
    train = [tf.train.AdamOptimizer(0.001).minimize(x) for x in model.xent]

with tf.Session(graph=model.g) as sess:
    def epoch_hook(epoch):
        if epoch%3 == 0:
            print('Train @t={}'.format(epoch))
            print(report(model, report_gen, sess))

    init = tf.global_variables_initializer()
    sess.run(init)

    for batch in train_gen.gen_batch_epochs(nepochs, epoch_hook):
        sess.run(train, feed_dict=model.food(batch))

    print('Vadation:')
    print(report(model, val_gen, sess))
