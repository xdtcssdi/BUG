from Activation import *


def activation(Z):

    A = Leak_Relu(Z)
    print("A = ", A)
    grad = Leak_ReluGrad(Z)
    print("Relugra = ",grad)


if __name__ == "__main__":
    Z = np.random.randn(10, 10)
    activation(Z)
