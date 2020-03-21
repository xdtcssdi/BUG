from BUG.function.Loss import SoftCategoricalCross_entropy, CrossEntry
from BUG.load_package import p


# 多输出评估
def evaluate_many(A, Y_train, layers):
    for layer in layers:
        A = layer.forward(A, None, mode='test')
    loss = SoftCategoricalCross_entropy().forward(Y_train, A)
    return loss, (p.argmax(A, -1) == Y_train).sum() / A.shape[0]


# 单输出评估
def evaluate_one(A, Y_train, layers):
    for layer in layers:
        A = layer.forward(A, None, mode='test')
    loss = CrossEntry().forward(Y_train, A)
    return loss, ((A > 0.5) == Y_train).sum() / A.shape[0]


if __name__ == '__main__':
    A = p.array([[0, 0.4, 0.3],
                 [1, 0.4, 0.1],
                 [0, 0.2, 0.6]])
    B = p.array([1, 0, 1])

    print((p.argmax(A, -1) == B).sum() / A.shape[0])
