import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import convert_to_tensor
from tensorflow.data import Dataset
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from witwidget.notebook.visualization import WitConfigBuilder, WitWidget

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

dat_df.head()

x_train, x_heldout, y_train, y_heldout = train_test_split(
    dat_df.iloc[:, :-1], dat_df.iloc[:, -1], test_size=0.2, stratify=dat_df["class"]
)

model = Sequential()
model.add(Dense(1, input_shape=(x_train.shape[1],), activation="sigmoid"))
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy", "AUC"]
)

n_cv_splits = 10

for slice_num, (train_ids, test_ids) in enumerate(
    StratifiedKFold(n_splits=n_cv_splits).split(x_train, y_train), 1
):
    slice_train_xs, slice_test_xs = x_train.iloc[train_ids], x_train.iloc[test_ids]
    slice_train_ys, slice_test_ys = y_train.iloc[train_ids], y_train.iloc[test_ids]

    model.fit(x=slice_train_xs, y=slice_train_ys, epochs=10, verbose=0)

    model.evaluate(x=slice_test_xs, y=slice_test_ys)

model.evaluate(x=x_heldout, y=y_heldout)

y_true = y_heldout.values
y_prediction_probabilities = model.predict(x_heldout)

roc_curve_df = (
    pd.DataFrame(roc_curve(y_true, y_prediction_probabilities))
    .T.rename(columns={0: "FPR", 1: "TPR", 2: "threshold"})
    .drop(index=0)
    .reset_index(drop=True)
)
roc_curve_df["Y"] = roc_curve_df["TPR"] - roc_curve_df["FPR"]
youden_T = roc_curve_df.iloc[roc_curve_df["Y"].idxmax()]["threshold"]

y_pred_youden = np.where(y_prediction_probabilities >= youden_T, 1, 0)

heldout_perf_roc = roc_auc_score(y_true, y_prediction_probabilities)
heldout_perf_prc = average_precision_score(y_true, y_prediction_probabilities)

confusion_mat = confusion_matrix(y_true, y_pred_youden)
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
perf_metrics
