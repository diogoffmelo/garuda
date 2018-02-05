import tensorflow as tf

from .base import Layer, OuputLayer
from util.layers import full_layer, xentropy_loss, char_accuracy, word_accuracy


class LinearSingleCharOutputLayer(OuputLayer):
    def __init__(self, ichar, graph, name):
        OuputLayer.__init__(self, graph, name)
        self.ichar = ichar
        self.name = '{}_{}'.format(self.name, ichar)
    
    def build(self, other):
        OuputLayer.build(self, other)
        with self.g.as_default(), tf.name_scope(self.name):
            self.cyin = tf.reshape(self.yin[:, self.ichar, :], 
                             [-1, self.yin.shape[-1]], name='select')

            self.shape = [int(self.xin.shape[-1]), int(self.cyin.shape[-1])]            
            self.fl = full_layer(self.xin, self.shape, self.bname('fl'), None)
            self.ylogits = self.fl['aout']
            self.ypreds = tf.nn.softmax(self.ylogits)
            self.accl = char_accuracy(
                            self.cyin, 
                            self.ypreds, 
                            self.bname('acc'))
            self.xentl = xentropy_loss(
                            self.ylogits, 
                            self.cyin, 
                            self.bname('xent'))

        self.cacc = self.accl['cacc']
        self.cpred = self.accl['cpred']
        self.loss = self.xentl['loss']
        self.xent = self.xentl['xent']
        self.vars += [self.fl['W'], self.fl['b']]
        
        self.summ += self.xentl['summary'] + self.accl['summary']
        self.summ_ext += self.fl['summary']
        self.metrics = {
            'cacc': self.cacc,
            'loss': self.loss
        }


class LinearLayer(Layer):
    def __init__(self, ishape, graph, name):
        Layer.__init__(self, graph, name)
        self.ishape = ishape

    def build(self, other):
        Layer.build(self, other)
        with self.g.as_default():
            self.fl = full_layer(self.xin, self.ishape, self.bname('fl'), tf.nn.relu)

        self.xout = self.fl['aout']
        self.vars += [self.fl['W'], self.fl['b']]
        self.summ_ext += self.fl['summary']


class LinearMultiCharOutputLayer(OuputLayer):
    def __init__(self, nchars, graph, name):
        OuputLayer.__init__(self, graph, name)
        self.nchars = nchars
        self.cmodels = [LinearSingleCharOutputLayer(i, graph, name)
                                for i in range(nchars)]

    def build(self, other):
        OuputLayer.build(self, other)
        self.cacc = []
        self.cpred = []
        self.loss = []
        self.xent = []
        for cmodel in self.cmodels:
            cmodel.build(other)
            self.vars += cmodel.vars
            self.cacc.append(cmodel.cacc)
            self.cpred.append(cmodel.cpred)
            self.loss.append(cmodel.loss)
            self.xent.append(cmodel.xent)
            self.summ += cmodel.summ
            self.summ_ext += cmodel.summ_ext

        with self.g.as_default():
            self.wacc = word_accuracy(self.cpred, self.cacc, self.bname('wacc'))

        self.summ = self.wacc['summary']
        self.metrics = {
            'cacc': self.cacc,
            'loss': self.loss,
            'pacc': self.wacc['pacc'], 
            'wacc': self.wacc['wacc'],
        }
