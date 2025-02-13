import numpy as np
from torch import Tensor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def CalEvalution(y_true,y_pred):
    mse = round(mean_squared_error(y_true, y_pred),4)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)),4)
    mae = round(mean_absolute_error(y_true, y_pred),4)
    evs = round(explained_variance_score(y_true, y_pred),4)
    r2 = r2_score(y_true, y_pred)
    return mse,rmse,mae,r2,evs

def sample_data(x, y, sample_proportion):
    indices = np.random.choice(len(x), size=int(len(x) * sample_proportion), replace=False)
    return x[indices], y[indices]

def test_reg_func(x,y,sample_time,sample_proportion):

    mse_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    evs_list = []

    for i in range(sample_time):
        np.random.seed(i)
        x_sample, y_sample = sample_data(x, y, sample_proportion)
        mse,rmse, mae, r2, evs = CalEvalution(y_sample, x_sample)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        evs_list.append(evs)

    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    r2_mean = np.mean(r2_list)
    r2_std = np.std(r2_list)
    evs_mean = np.mean(evs_list)
    evs_std = np.std(evs_list)

    return mse_mean, mse_std,rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std




