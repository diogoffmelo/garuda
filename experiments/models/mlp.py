import tensorflow as tf

from .base import BaseModel
from .linear import LinearW, CopyMeticsLayerMixin, LinearFoodMixin
from util.layers import full_layer


class Mlp(LinearFoodMixin, CopyMeticsLayerMixin, BaseModel):
    def __init__(self, xin, yin, nint, graph, name):
        BaseModel.__init__(self, xin, yin, graph, name)
        self.shape = [self.xshape[-1], nint]
        with self.g.as_default():
            self.fl = full_layer(self.xin, self.shape, self.bname('fl'), tf.nn.relu)
        
        self.lin = LinearW(self.fl['aout'], yin, graph, self.bname('lin'))
        self.copy_metrics(self.lin)
        self.vars += [self.fl['W'], self.fl['b']]