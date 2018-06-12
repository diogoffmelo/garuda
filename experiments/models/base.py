import numpy as np
import tensorflow as tf


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
        self.xin = other.xout
        self.yin = other.yout
        self.yout = self.yin
        self.xout = self.yin


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
        self.yout = self.yin

    def food(self, batch):
        _X, _Y = batch
        return {self.yin: _Y, self.xin: _X}


    def build(self, other):
        raise NotImplementedError()


class OuputLayer(Layer):
    pass


class ReshapeLayer(Layer):
    def __init__(self, outshape, graph, name):
        Layer.__init__(self, graph, name)
        self.outshape = list(map(int, outshape)) if outshape else []


    def build(self, other):
        Layer.build(self, other)
        with self.g.as_default(), tf.name_scope(self.bname('reshape')):
            self.xout = tf.reshape(self.xin, 
                                   self.outshape,
                                   name='reshape')


class LinearReshapeLayer(ReshapeLayer):
    def __init__(self, graph, name):
        ReshapeLayer.__init__(self, None, graph, name)


    def build(self, other):
        _, x1 ,x2, x3 = other.xout.shape
        self.outshape = [-1, int(x1)*int(x2)*int(x3)]
        ReshapeLayer.build(self, other)


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
