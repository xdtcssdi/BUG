class Layer(object):

    def __init__(self, unit_number=0, activation="relu"):
        self.unit_number = unit_number
        self.activation = activation
        self.pre_layer = None
        self.next_layer = None
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.isFirst = False
        self.isLast = False
        self.batchNormal = None
        self.A_pre = None
        self.Z = None

    def init_params(self, nx):
        raise NotImplementedError

    def forward(self, A_pre, mode='train'):
        raise NotImplementedError

    def backward(self, pre_grad):
        raise NotImplementedError
