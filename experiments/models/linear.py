import tensorflow as tf

from .base import BaseModel
from util.layers import full_layer, xentropy_loss, char_accuracy, word_accuracy


class LinearC(BaseModel):
    def __init__(self, xin, yin, graph, name):
        BaseModel.__init__(self, xin, yin, graph, name)
        self.shape = [self.xshape[-1], self.yshape[-1]]
        with self.g.as_default():
            self.fl = full_layer(self.xin, self.shape, self.bname('fl'), None)
            self.ylogits = self.fl['aout']
            self.ypreds = tf.nn.softmax(self.ylogits)
            self.accl = char_accuracy(
                            self.yin, 
                            self.ypreds, 
                            self.bname('acc'))
            self.xentl = xentropy_loss(
                            self.ylogits, 
                            self.yin, 
                            self.bname('xent'))


        self.cacc = self.accl['cacc']
        self.cpred = self.accl['cpred']
        self.loss = self.xentl['loss']
        self.xent = self.xentl['xent']
        self.vars += [self.fl['W'], self.fl['b']]


class LinearW(BaseModel):
    def __init__(self, xin, yin, graph, name):
        BaseModel.__init__(self, xin, yin, graph, name)
        self.nchars, self.nclasses = self.yshape
        self.cmodels = []
        self.cacc = []
        self.cpred = []
        self.loss = []
        self.xent = []
        for i in range(self.nchars):
            with self.g.as_default():
                cyi = tf.reshape(self.yin[:, i, :], [-1, self.nclasses])
            
            cmodel = LinearC(xin, cyi, graph, self.bname('cm{}'.format(i)))
            self.cmodels.append(cmodel)
            self.vars += cmodel.vars
            self.cacc.append(cmodel.cacc)
            self.cpred.append(cmodel.cpred)
            self.loss.append(cmodel.loss)
            self.xent.append(cmodel.xent)

        with self.g.as_default():
            self.wacc = word_accuracy(self.cpred, self.cacc, self.bname('wacc'))

        self.metrics = {
            'cacc': self.cacc,
            'loss': self.loss,
            'pacc': self.wacc['pacc'], 
            'wacc': self.wacc['wacc'],
        }


    def food(self, batch):
        _X, _Y = batch
        return {self.yin: _Y, self.xin: _X.reshape([-1, self.xshape[-1]])}
