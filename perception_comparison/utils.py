import gin
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pystan
import scipy.stats
import sys
from enum import Enum
from scipy.special import expit as calc_sigmoid
from scipy.stats import ks_2samp

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update(
    {'font.size': 14,
     'pdf.fonttype' : 42})
plt.rcParams['pdf.fonttype'] = 42

SUBGROUPS = ['microcalcifications', 'soft_tissue_lesions', 'mixed', 'occult', 'nonbiopsied']
SEVERITIES = ['0', '0.17', '0.22', '0.33', '0.5', '0.67', '1', '2', '4']

class Side(Enum):
    LEFT = 0
    RIGHT = 1

def save_file(obj, fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def get_subgroups(exam_info):
    subgroups = exam_info[['subgroup_l', 'subgroup_r']].copy()
    for subgroup_name in SUBGROUPS:
        subgroups.replace(subgroup_name, SUBGROUPS.index(subgroup_name), inplace=True)
    subgroups = subgroups.values
    return subgroups

def get_exam_idxs(exam_info, query):
    is_subtask = 'subtask' in query
    subgroup = query.replace(' subtask', '')
    idxs_l, idxs_r = [], []
    for exam_idx in range(len(exam_info)):
        if subgroup in exam_info.subgroup_l.iloc[exam_idx] or (is_subtask and exam_info.y_l.iloc[exam_idx] != 'malignant'):
            idxs_l.append(exam_idx)
        if subgroup in exam_info.subgroup_r.iloc[exam_idx] or (is_subtask and exam_info.y_r.iloc[exam_idx] != 'malignant'):
            idxs_r.append(exam_idx)
    return idxs_l, idxs_r

def get_y(exam_info, exam_idxs_l, exam_idxs_r):
    y = np.concatenate((exam_info.y_l.iloc[exam_idxs_l].values, exam_info.y_r.iloc[exam_idxs_r].values))
    y = (y == 'malignant').astype(int)
    return y

def calc_posterior_pred_elem(mu, gamma, nu, b, side, exam_idx, reader_idx, severity_idx, category_idx):
    pred = []
    np.random.shuffle(mu[exam_idx, side])
    np.random.shuffle(gamma[severity_idx, category_idx])
    np.random.shuffle(nu[reader_idx, category_idx])
    np.random.shuffle(b[category_idx])
    for mu_s, gamma_s, nu_s, b_s in zip(mu[exam_idx, side], gamma[severity_idx, category_idx], nu[reader_idx,
            category_idx], b[category_idx]):
        pred.append(calc_sigmoid(mu_s + gamma_s + nu_s + b_s))
    pred = np.mean(pred)
    return pred

def calc_posterior_pred(exam_info_fpath, pgm_fpath):
    mu, gamma_, gamma, nu, b = load_file(pgm_fpath)
    n_exams, n_readers, n_severities = mu.shape[0], nu.shape[0], gamma.shape[0]
    subgroups = get_subgroups(pd.read_csv(exam_info_fpath))
    pred = np.full((n_readers, n_severities, 2 * n_exams), np.nan)
    for reader_idx in range(n_readers):
        for severity_idx in range(n_severities):
            pred_l, pred_r = [], []
            for exam_idx_l in range(n_exams):
                pred_l.append(calc_posterior_pred_elem(mu, gamma, nu, b, Side.LEFT.value, exam_idx_l, reader_idx,
                    severity_idx, subgroups[exam_idx_l, Side.LEFT.value]))
            for exam_idx_r in range(n_exams):
                pred_r.append(calc_posterior_pred_elem(mu, gamma, nu, b, Side.RIGHT.value, exam_idx_r, reader_idx,
                    severity_idx, subgroups[exam_idx_r, Side.RIGHT.value]))
            pred[reader_idx, severity_idx] = np.concatenate((pred_l, pred_r))
    return pred

def split_arr(x):
    assert len(x) % 2 == 0
    half_idx = len(x) // 2
    return x[:half_idx], x[half_idx:]

def tight_layout(fig):
    for _ in range(5):
        fig.tight_layout(pad=2)