import numpy as np
import tensorflow as tf

class BaseModel(object):
    def __init__(self, xin, yin, graph, name):
        self.g = graph
        self.xin = xin
        self.yin = yin
        self.xshape = [int(d) for d in xin.shape[1:]]
        self.yshape = [int(d) for d in yin.shape[1:]]
        self.name = name
        self.vars = []
        self.metrics = {}


    def bname(self, comp):
        return '{}.{}'.format(self.name, comp)


class Layer(object):
    def __init__(self, graph, name):
        self.g = graph
        self.name = name
        self.vars = []
        self.summ = []
        self.summ_ext = []

    def bname(self, comp):
        return '{}.{}'.format(self.name, comp)

    def build(self, other):
        raise NotImplementedError()


class InputLayer(Layer):
    def __init__(self, xshape, yshape, graph, name):
        Layer.__init__(self, graph, name)
        with graph.as_default(), tf.name_scope(self.bname('placeholders')):
            self.xin = tf.placeholder(tf.float32, 
                                      xshape, 
                                      name='X')
            self.yin = tf.placeholder(tf.float32, 
                                   yshape, 
                                   name='Y')

        self.xout = self.xin

    def food(self, batch):
        _X, _Y = batch
        return {self.yin: _Y, self.xin: _X}


class OuputLayer(Layer):
    pass


class LinearInputLayer(InputLayer):
    def __init__(self, xshape, yshape, graph, name):
        InputLayer.__init__(self, xshape, yshape, graph, name)
        x1, x2, x3 = list(map(int, self.xin.shape[1:]))
        with graph.as_default(), tf.name_scope(self.bname('reshape')):
            self.xout = tf.reshape(self.xin, 
                                   [-1, x1 * x2 * x3],
                                   name='reshape')


class StackedLayers(object):
    def __init__(self, *layers):
        assert isinstance(layers[0], InputLayer)
        assert isinstance(layers[-1], OuputLayer)
        for l1, l2 in zip(layers[:-1], layers[1:]):
            l2.build(l1)
            l2.summ = list(set(l2.summ + l1.summ))
            l2.summ_ext = list(set(l1.summ_ext + l1.summ_ext))

        layers[-1].food = layers[0].food
        self.model = layers[-1]





