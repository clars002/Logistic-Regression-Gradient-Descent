# This file contains functions pertaining to model evaluation, namely ROC curve generation and constituent functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logistic_regression import predict


# Calculate TPR
def tpr(predictions, labels):
    correct_predictions = predictions[predictions == labels]
    true_positives = len(correct_predictions[correct_predictions == 1])
    actual_positives = len(labels[labels == 1])

    return true_positives / actual_positives


# Calculate FPR
def fpr(predictions, labels):
    incorrect_predictions = predictions[predictions != labels]
    false_positives = len(incorrect_predictions[incorrect_predictions == 1])
    actual_negatives = len(labels[labels == 0])

    return false_positives / actual_negatives


def confusion_matrix(predictions, labels):
    correct_predictions = predictions[predictions == labels]
    true_positives = len(correct_predictions[correct_predictions == 1])

    incorrect_predictions = predictions[predictions != labels]
    false_positives = len(incorrect_predictions[incorrect_predictions == 1])

    true_negatives = len(correct_predictions[correct_predictions == 0])
    false_negatives = len(incorrect_predictions[incorrect_predictions == 0])

    conf_mat = pd.DataFrame(
        [[true_positives, false_negatives], [false_positives, true_negatives]],
        columns=["Positive", "Negative"],
        index=["Positive", "Negative"],
    )
    conf_mat.columns.name = "Predicted Class"
    conf_mat.index.name = "Actual Class"

    return conf_mat


# Generate/plot ROC curve
def roc_curve(p_vals: np.array, labels: np.array, step: int = 1e-5, plot: bool = True):
    thresholds = np.arange(0, 1, step)
    num_pts = len(thresholds)
    tprs = [0] * num_pts
    fprs = [0] * num_pts
    for i in range(num_pts):
        prd = predict(p_vals, thresholds[i])
        tprs[i] = tpr(prd, labels)
        fprs[i] = fpr(prd, labels)

    if plot:
        plt.figure(1)
        plt.plot(fprs, tprs)
        plt.scatter(fprs, tprs, marker="o", alpha=0.5)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

    return tprs, fprs
