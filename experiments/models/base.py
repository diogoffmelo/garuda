import numpy as np


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


    def food(self, batch):
        raise NotImplementedError()
