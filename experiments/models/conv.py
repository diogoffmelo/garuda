import tensorflow as tf

from .base import Layer
from util.layers import conv_layer


class ConvLayer(Layer):
    def __init__(self, shape, stride, graph, name):
        Layer.__init__(self, graph, name)
        self.shape = shape
        self.stride = stride


    def build(self, other):
        Layer.build(self, other)
        layer_args = {
            'stridep': self.stride, 
            'activation': tf.nn.relu
        }
        with self.g.as_default(), tf.name_scope(self.name):
            self.conv  = conv_layer(self.xin,
                                    self.shape,
                                    self.bname('conv'), 
                                    **layer_args)


        self.xout = self.conv['aout']
        self.summ += []
        self.summ_ext += self.conv['summary']
