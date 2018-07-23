import tensorflow as tf

from .base import Layer
from util.layers import conv_layer, conv_layer_max


class ConvLayer(Layer):
    def __init__(self, shape, stride, graph, name, maxpool=False):
        Layer.__init__(self, graph, name)
        self.shape = shape
        self.stride = stride
        self.maxpool = maxpool


    def build(self, other):
        Layer.build(self, other)
        layer_args = {
            'stridep': self.stride, 
            'activation': tf.nn.relu
        }
        with self.g.as_default(), tf.name_scope(self.name):
            if self.maxpool:
                self.conv  = conv_layer_max(self.xin,
                                            self.shape,
                                            self.bname('conv_max'), 
                                            **layer_args)

            else:
                self.conv  = conv_layer(self.xin,
                                        self.shape,
                                        self.bname('conv'), 
                                        **layer_args)


        self.xout = self.conv['aout']
        self.summ += []
        self.summ_ext += self.conv['summary']

    def num_parameters(self):
        _prod = 1
        for c in self.shape:
            _prod *= c

        return _prod

