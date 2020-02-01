import numpy as np


def CrossEntry(Y_train,Y_hat):

    cost = - Y_train * np.log(Y_hat) - (1-Y_train) * np.log(1-Y_hat)

    J = np.mean(cost, axis=1, keepdims=True)

    return np.squeeze(J)


def CrossEntryGrad(Y_train, Y_hat):

    return np.squeeze(-Y_train/Y_hat + (1-Y_train)/(1-Y_hat))