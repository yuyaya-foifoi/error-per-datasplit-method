import itertools

import numpy as np
import pandas as pd

random_state = np.random.RandomState(42)


def random_split(rating_data, test_size):
    permuted_index = random_state.permutation(len(rating_data.index))

    train_idxs, val_idxs, test_idxs = np.split(
        permuted_index,
        [
            int(len(permuted_index) * (1 - test_size * 2)),
            int(len(permuted_index) * (1 - test_size)),
        ],
    )

    train = rating_data.loc[rating_data.index.isin(train_idxs)]
    val = rating_data.loc[rating_data.index.isin(val_idxs)]
    test = rating_data.loc[rating_data.index.isin(test_idxs)]

    return train, val, test


def per_user_split(rating_data, test_size, col_name="UserID"):
    def get_train_val_test_index(usr_df: pd.DataFrame) -> pd.Series:
        index = usr_df.index

        train_idxs, val_idxs, test_idxs = np.split(
            index,
            [
                int(len(index) * (1 - test_size * 2)),
                int(len(index) * (1 - test_size)),
            ],
        )
        return pd.Series(
            {
                "train_idxs": train_idxs,
                "val_idxs": val_idxs,
                "test_idxs": test_idxs,
            }
        )

    ids = rating_data.groupby(col_name).apply(get_train_val_test_index)

    train_row_idxs = itertools.chain.from_iterable(ids.train_idxs)
    val_row_idxs = itertools.chain.from_iterable(ids.val_idxs)
    test_row_idxs = itertools.chain.from_iterable(ids.test_idxs)

    train = rating_data.loc[rating_data.index.isin(train_row_idxs)]
    val = rating_data.loc[rating_data.index.isin(val_row_idxs)]
    test = rating_data.loc[rating_data.index.isin(test_row_idxs)]

    return train, val, test
