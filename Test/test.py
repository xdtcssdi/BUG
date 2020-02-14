import numpy as np

from util import load_CIFAR10, one_hot
def predict(X_train, Y_train):
    p = .0
    for i in range(X_train.shape[0]):
        t1 = returnMaxIdx(X_train[i])
        t2 = returnMaxIdx(Y_train[i])
        if t1 == t2:
            p += 1
    print(p)
    print("accuracy: %f%%" % (p * 1.0 / X_train.shape[0] * 100.))
def returnMaxIdx(a):
    list_a = a.tolist()
    max_index = list_a.index(max(list_a))  # 返回最大值的索引
    return max_index
if __name__ == '__main__':

    x = np.array([
        [0.3,0.5,0.1],[0.5, 0.4,0.],[0.2,0.1,0.9]
    ])
    y = np.array([
        [0.,1.,0.],[1., 0.,0.],[0.,0.,1.]
    ])
    print(x)
    print(y)
    mask = np.argmax(x, axis=1)
    x[range(3), mask] = 1



