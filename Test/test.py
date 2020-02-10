from Layers.Convolution import Convolution
import numpy as np

class T:
    def __init__(self):
        self.a = None
        self.b = None
    def f(self):
        print(self.a)
    @property
    def params(self):
        return self.a, self.b

if __name__ == '__main__':
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    con = Convolution(8, (2, 2), stride=1, padding=2)
    con.forward(A_prev)
    con.backward(con.Z)
    print(con.Z.shape)
    print(con.dW.mean())
