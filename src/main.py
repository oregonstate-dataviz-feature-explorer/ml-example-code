#!/usr/bin/env python3
"""Build and test a logistic regression model"""

import os
import multiprocessing
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, model_selection
from sys import argv

dat_in_pth = "../data/features.pkl"
dat_df = pd.read_pickle(dat_in_pth)

results_dir = "results"
seed = 8675309
np.random.seed(seed)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

training_set_tss_ids_pth = results_dir + "/training_set_tss_ids.txt"
heldout_test_set_tss_ids_pth = results_dir + "/heldout_test_set_tss_ids.txt"
crossval_results_pth = results_dir + "/crossval_results.csv"
test_performance_pth = results_dir + "/heldout_test_performance.csv"
coefs_pth = results_dir + "/coefficients.csv"

###########################################################################
# split data into 80/20 train/test
train, heldout_test = model_selection.train_test_split(
    dat_df, test_size=0.2, stratify=dat_df["class"], random_state=seed
)

np.savetxt(training_set_tss_ids_pth, train.index.values, fmt="%s")
np.savetxt(heldout_test_set_tss_ids_pth, heldout_test.index.values, fmt="%s")

###########################################################################
# define log reg cv object
n_procs = 10

log_cv = linear_model.LogisticRegressionCV(
    solver="liblinear",
    Cs=np.logspace(-4, 4, 100),
    cv=5,
    penalty="l1",
    n_jobs=n_procs,
    refit=True,
    scoring="average_precision",
    random_state=seed,
)

# train model
log_cv.fit(train.iloc[:, :-1], y=train["class"])

###########################################################################
# test final model on heldout test set
y_true, y_prediction_probabilities = (
    heldout_test["class"],
    log_cv.predict_proba(heldout_test.iloc[:, :-1]),
)

###########################################################################
# get various metrics and save to disk
roc_curve_df = (
    pd.DataFrame(metrics.roc_curve(y_true, y_prediction_probabilities[:, 1]))
    .T.rename(columns={0: "FPR", 1: "TPR", 2: "threshold"})
    .drop(index=0)
    .reset_index(drop=True)
)
roc_curve_df["Y"] = roc_curve_df["TPR"] - roc_curve_df["FPR"]
youden_T = roc_curve_df.iloc[roc_curve_df["Y"].idxmax()]["threshold"]

y_pred_youden = np.where(y_prediction_probabilities[:, 1] >= youden_T, 1, 0)

heldout_perf_roc = metrics.roc_auc_score(y_true, y_prediction_probabilities[:, 1])
heldout_perf_prc = metrics.average_precision_score(
    y_true, y_prediction_probabilities[:, 1]
)

confusion_mat = metrics.confusion_matrix(y_true, y_pred_youden)
TN = confusion_mat[0][0]
FP = confusion_mat[0][1]
FN = confusion_mat[1][0]
TP = confusion_mat[1][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
# Specificity or true negative rate
TNR = TN / (TN + FP)
# Precision or positive predictive value
PPV = TP / (TP + FP)

heldout_perf_f1 = 2 * ((PPV * TPR) / (PPV + TPR))

perf_metrics = pd.Series(
    {
        "C": log_cv.C_[0],
        "auROC": heldout_perf_roc,
        "auPRC": heldout_perf_prc,
        "youden_T": youden_T,
        "sensitivity": TPR,
        "specificity": TNR,
        "precision": PPV,
        "f1": heldout_perf_f1,
        "seed": seed,
    }
)

coefficients = pd.DataFrame(log_cv.coef_, columns=dat_df.columns[:-1])

# save metrics
perf_metrics.to_csv(test_performance_pth, header=True)

# save coefficient values
coefficients.to_csv(coefs_pth, header=True)
