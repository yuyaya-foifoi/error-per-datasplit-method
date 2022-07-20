import numpy as np
from sklearn.metrics import mean_squared_error


def _fit(model, train_data):
    model.fit(train_data)
    Theta = model.transform(train_data)
    M = model.components_.T
    return M, Theta


def _predict(M, Theta):
    pred = M.dot(Theta.T)
    pred = pred.T
    return pred


def get_error(model, train_data, eval_data):
    M, Theta = _fit(model, train_data)
    pred = _predict(M, Theta)
    return mean_squared_error(pred.flatten(), eval_data.flatten()) / np.sum(
        eval_data > 0
    )
