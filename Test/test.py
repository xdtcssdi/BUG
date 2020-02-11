from Layers.Convolution import Convolution
import numpy as np

class T:
    def __init__(self):
        self.a = None
        self.b = None
    def f(self):
        print(self.a)

def predict(X_train, Y_train):
    p = .0
    for i in range(X_train.shape[0]):
        t1 = X_train[i][0] > 0.5
        t2 = Y_train[i][0]
        if t1 == t2:
            p += 1
    print("accuracy: %.1f%%" % (p/X_train.shape[0]*100.))


if __name__ == '__main__':
    a1 = np.random.randn(209, 1)
    a2 = np.random.randn(209, 20)
    print((np.mean(a2,axis=1)).shape)