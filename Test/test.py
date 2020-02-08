from Activation import *


def activation(Z):

    A = Leak_Relu(Z)
    print("A = ", A)
    grad = Leak_ReluGrad(Z)
    print("Relugra = ",grad)


def tt(*d):
    print(d)


def forward(input):
    x = input - np.max(input, axis=1, keepdims=True)
    exp_x = np.exp(x)
    s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return s

def derivative(self, input=None):
    last_forward = input if input else self.last_forward
    return np.ones(last_forward.shape)


if __name__ == "__main__":
    x = np.random.randn(2, 2)
    print(forward(x))