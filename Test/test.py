from Activation import *


def activation(Z):

    A = Leak_Relu(Z)
    print("A = ", A)
    grad = Leak_ReluGrad(Z)
    print("Relugra = ",grad)


def tt(*d):
    print(d)


if __name__ == "__main__":
    print(tt(*(1,2,3)))