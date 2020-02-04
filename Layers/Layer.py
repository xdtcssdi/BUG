import numpy as np
class Layer(object):

    def __init__(self, unit_number,  activation="relu"):
        self.unit_number = unit_number
        self.activation = activation

    def init_params(self, nx):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def backward(self, pre_grad, pre_W=None):
        raise NotImplementedError
