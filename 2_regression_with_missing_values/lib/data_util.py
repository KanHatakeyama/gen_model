# utility funcs to process data

import numpy as np
import pandas as pd
import copy
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# prepare dataframe with randon NaN values


def prep_nan_df(df, threshold, target_id):
    """
    df: dataframe
    threshold (0-1): if 0.3, 30% of the data will be filled with NaN randomly
    target_id: column ID of y (i.e., y won't be filled with NaN)
    return: dataframe with NaN
    """
    array = np.array(df)
    random_mask = np.random.rand(array.shape[0], array.shape[1])
    random_mask = random_mask > threshold

    random_mask[:, target_id] = 1

    array[~random_mask] = np.nan
    nan_df = pd.DataFrame(array)
    nan_df.columns = df.columns
    return nan_df

# split dataframe into train and test dataframe


def prepare_dataset(df, target, spl_ratio):
    """
    df: dataframe
    target: target column name (string) of y
    spl_ratio: split ratio. if 0.7, 70% of the data will be used as train
    return: (dataframe of train dataset), (dataframe of ORIGINAL DATABASE)
            (NOTE: target value of test dataset will be filled with NaN just before regression, not in this function)
    """
    sel_df = df.dropna(how='all', axis=1)
    sel_df = sel_df.loc[:, (sel_df != 0).any(axis=0)]
    spl_pos = int((spl_ratio)*sel_df.shape[0])

    te_df = sel_df
    tr_df = copy.copy(te_df[:spl_pos])

    return tr_df, te_df

# prepare numpy arrays for imputation


def prepare_data_arrays(tr_df, te_df, target):
    """
    tr_df: train dataset made by "prepare_dataset" function
    te_df: test dataset made by "prepare_dataset" function
    target: name of target y
    return: (numpy array of train dataset),
            (numpy array of test dataset: y will be filled with NaN), 
            (column ID of y)
    """
    col_to_id = {k: v for v, k in enumerate(tr_df.columns)}
    train_array = np.array(tr_df)
    test_array = np.array(te_df)

    target_id = col_to_id[target]

    # fill target values with nan
    test_array[:, target_id] = np.nan

    return train_array, test_array, target_id


# visualize regression result
def visualize_result(tr_y, pred_tr_y, te_y, pred_te_y):
    """
    tr_y: actual y for train
    pred_tr_y: predicted y for train
    te_y: actual y for test
    pred_te_y: predicted y for test
    """
    ax = plt.gca()
    ax.scatter(tr_y, pred_tr_y)
    ax.scatter(te_y, pred_te_y)
    ax.set(ylim=(0, 1), ylabel='actual', xlim=(
        0, 1), xlabel='predicted', aspect='equal')

    print("MAE for train: ", mean_absolute_error(tr_y, pred_tr_y))
    print("MAE for test: ", mean_absolute_error(te_y, pred_te_y))
