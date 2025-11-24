# Misc/utilities.py
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

def get_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def df_empty(columns, dtypes):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame()
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

def binary_one_hot(labels, device):
    y_onehot = torch.FloatTensor(len(labels), 2).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels, 1)
    return y_onehot

def calculate_test_log(pred_prob, label):
    assert len(pred_prob) == len(label)
    assert np.max(label) <= 1
    assert np.min(label) >= 0
    pred_class = np.where(pred_prob > 0.5, 1, 0)
    metrics = {}
    TP = np.sum(pred_class * label)
    TN = np.sum((pred_class - 1) * (label - 1))
    FP = np.sum(np.abs(pred_class * (label - 1)))
    FN = np.sum(np.abs((pred_class - 1) * label))
    
    metrics["Accy"] = (TP + TN) / len(label)
    metrics["Prec"] = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    metrics["Recall"] = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    metrics["MCC"] = (TP * TN - FP * FN) / denom if denom > 0 else np.nan
    metrics["Up Ratio"] = np.mean(label)
    metrics["Pred Up Ratio"] = np.mean(pred_class)
    
    return metrics

def rank_corr(df, col1, col2, method="spearman"):
    if method == "spearman":
        return df[col1].rank().corr(df[col2].rank())
    elif method == "pearson":
        return df[col1].corr(df[col2])
    else:
        raise ValueError(f"Method {method} not supported")

def to_latex_w_turnover(df, cut=10):
    new_idx = ["Low"] + list(range(2, cut)) + ["High", "H-L"]
    new_idx_map = dict(zip(range(cut + 1), new_idx))
    df_copy = df.copy()
    df_copy.index = [new_idx_map.get(i, i) for i in df_copy.index]
    return df_copy.to_latex()

def save_pkl_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

import torch