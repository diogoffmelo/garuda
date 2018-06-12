import numpy as np
import tensorflow as tf

from util.batch import NumpyBatchGenerator, SampleMode
from util.load import load_to_batches


TEST = False
model_path = './graphs/linearw/'
model_path_ckpts = model_path + 'ckpts/epoch' 
model_path_boards = model_path

if TEST:
    nepochs = 4
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }
else:
    nepochs = 30
    dbargs = {
        'path': '../generate/data.hdf5',
    }
    batchargs = {
        'train_bsize': 100,
        'report_bsize': 100,
        'val_bsize': 100,
    }

train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)

class Report(object):
    def __init__(self, batch_gen, sess):
        self.batch_gen = batch_gen
        self.size = batch_gen.set_size
        self.sess = sess
        self._reset()

    def _reset(self):
        self.nacc = 0
        self.naccs = np.zeros([5])

    def inc(self, sess_report):
        nacc, naccs = sess_report
        self.nacc += nacc
        self.naccs += np.asarray(naccs)

    def evaluate(self, nacc, naccs):
        for batch in self.batch_gen.gen_batches():
            __X, __Y = batch
            self.inc(self.sess.run([nacc, naccs], feed_dict={X: __X, Y: __Y}))

        print(self.nacc/self.size)
        print(self.naccs/self.size)
        self._reset()

train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)

tf.reset_default_graph()
with tf.Session(graph=tf.Graph()) as sess:
    name = lambda comp: '{}-{}'.format(model_path_ckpts, comp)
    saver = tf.train.import_meta_graph(name('0.meta'))
    saver.restore(sess, name(0))
    g = sess.graph
    X = g.get_tensor_by_name('input.placeholders/X:0')
    Y = g.get_tensor_by_name('input.placeholders/Y:0')
    wacc = g.get_tensor_by_name('classificador.wacc/wacc:0')
    nacc = g.get_tensor_by_name('classificador.wacc/nacc:0')
    pacc = g.get_tensor_by_name('classificador.wacc/pacc:0')
    naccs = [g.get_tensor_by_name('classificador_{0}/classificador_{0}.acc/ncacc:0'.format(i)) for i in range(5)]

    val_report = Report(val_gen, sess)
    train_report = Report(train_gen, sess)
    for i in range(nepochs):
        print('Report @t={}'.format(i))
        print('Train:')
        train_report.evaluate(nacc, naccs)
        print('Validation:')
        val_report.evaluate(nacc, naccs)
        saver.restore(sess, name(i+1))

    print('Report @t={}'.format(nepochs))
    print('Train:')
    train_report.evaluate(nacc, naccs)
    print('Validation:')
    val_report.evaluate(nacc, naccs)
