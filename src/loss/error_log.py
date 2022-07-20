import numpy as np
from sklearn.decomposition import NMF

from src.data.matrix import to_matrix
from src.loss.error import get_error

NMF_param = {
    "n_components": 30,
    "init": "random",
    "random_state": 0,
    "l1_ratio": 0,
    "max_iter": 30,
}


def get_log_per_test_size(
    split_type, ratings, TEST_SIZE=[0.05, 0.1, 0.15, 0.2]
):

    log = np.zeros((len(TEST_SIZE), 2))

    for i_idx, test_size in enumerate(TEST_SIZE):

        train, val, test = split_type(ratings, test_size)
        train = to_matrix(train)
        val = to_matrix(val)
        test = to_matrix(test)

        for j_idx, data in enumerate([val, test]):

            model = NMF(**NMF_param)
            log[i_idx, j_idx] = get_error(model, train, data)

    return log
