from BUG.function.Loss import SoftCategoricalCross_entropy, CrossEntry
from BUG.load_package import p


# 多输出评估
def evaluate_many(X_train, Y_train, layers):
    A = X_train
    for layer in layers:
        A = layer.forward(A, None, mode='test')
    loss = SoftCategoricalCross_entropy().forward(Y_train, A)
    return loss, (p.argmax(A, -1) == Y_train).sum() / X_train.shape[0]


# 单输出评估
def evaluate_one(A, Y_train, layers):
    for layer in layers:
        A = layer.forward(A, None, mode='test')
    loss = CrossEntry().forward(Y_train, A)
    return loss, ((A > 0.5) == Y_train).sum() / A.shape[0]