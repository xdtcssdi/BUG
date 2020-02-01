from Activation import *


def activation(Z):

    A = Leak_Relu(Z)
    print("A = ", A)
    grad = Leak_ReluGrad(Z)
    print("Relugra = ",grad)


class A:
    def __init__(self):
        self.B = 10

if __name__ == "__main__":
    a = A()
    print(a.B)
