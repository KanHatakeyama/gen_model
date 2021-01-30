"""
Imputation functions for regression
"""

from data_util import prepare_data_arrays
import numpy as np
from tqdm.notebook import tqdm
from MCFlowImputer import MCFlowImputer
from MissParallelX import MissParallelX
from sklearn.linear_model import SGDRegressor


# imputation by MCFlow
def imputation_MCFlow(tr_df, te_df, target, n_epochs=128, loss_ratio=100):
    """
    tr_df: train dataset made by "prepare_dataset" function
    te_df: test dataset made by "prepare_dataset" function
    target: name of target y
    n_epochs: number of epochs for imputation
    loss_ratio:  ratio of loss for flow model and reconstruction error. if 1>loss_ratio, the model will fit the training data more strongly.
    return: actual train y, predicted train y, actual test y, predicted test y
    """

    train_array, test_array, target_id = prepare_data_arrays(
        tr_df, te_df, target)

    if train_array.shape[0] > 5000:
        batch_size = 64
    else:
        batch_size = 8

    # imputation by MCFlow
    imputer = MCFlowImputer(
        n_epochs=n_epochs, batch_size=batch_size, verbose=False, loss_ratio=loss_ratio)
    res = imputer.fit_transform(train_array,
                                repeat=1,
                                target_id=target_id,
                                test_array=test_array,
                                )

    pred_y = res[1][:, target_id]
    ans_y = np.array(te_df[target])
    spl_pos = tr_df.shape[0]
    return ans_y[:spl_pos], pred_y[:spl_pos], ans_y[spl_pos:], pred_y[spl_pos:]


# imputation by missX
def imputation_missX(tr_df, te_df, target, imputer=MissParallelX(custom_regressor=SGDRegressor())):
    """
    tr_df: train dataset made by "prepare_dataset" function
    te_df: test dataset made by "prepare_dataset" function
    target: name of target y
    imputer: an imputer class, which has a "fit_transform" function for imputation
    return: actual train y, predicted train y, actual test y, predicted test y
    """
    # set data array
    train_array, test_array, target_id = prepare_data_arrays(
        tr_df, te_df, target)

    # impute missing values in train dataset
    imputed_train_array = imputer.fit_transform(train_array)

    # predict missing values and target value in test dataset
    pred_y_array = []

    # predictions are done for each record of test dataset, to avoid data leakeage among test data
    for i in tqdm(range(test_array.shape[0])):
        # input train data + one record of test (or train) data whose target is NaN
        temp_array = np.vstack((test_array[i], imputed_train_array))
        temp_array = imputer.fit_transform(temp_array)
        pred_y_array.append(temp_array[0, target_id])

    spl_pos = tr_df.shape[0]
    pred_y_array = np.array(pred_y_array)
    p_tr_y = pred_y_array[:spl_pos]
    p_te_y = pred_y_array[spl_pos:]

    ans_y = np.array(te_df[target])

    return ans_y[:spl_pos], p_tr_y, ans_y[spl_pos:], p_te_y
