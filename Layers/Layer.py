import numpy as np


class Layer(object):

    def __init__(self, unit_number,  activation="relu", input=None):
        self.unit_number = unit_number
        self.activation = activation
        self.pre_layer = None
        self.next_layer = None
        self.input = input
        self.dA = None
        self.dZ = None
        self.W = None
        self.b = None
        self.A = None
        self.dW = None
        self.db = None
        self.Z = None
        self.A = None
        self.isFirst = False
        self.isLast = False
        self.batchNormal = None
        self.beta = None
        self.gamma = None
        self.dbeta = None
        self.dgamma = None

    def init_params(self, nx):
        raise NotImplementedError

    def forward(self, input, mode='train'):
        raise NotImplementedError

    def backward(self, pre_grad):
        raise NotImplementedError

