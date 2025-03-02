import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import predict
from utils import make_color_gradient


def tpr(predictions, labels):
    actual_positives = len(labels[labels == 1])
    correct_predictions = predictions[predictions == labels]
    true_positives = len(correct_predictions[correct_predictions == 1])

    return true_positives / actual_positives


def fpr(predictions, labels):
    incorrect_predictions = predictions[predictions != labels]
    false_positives = len(incorrect_predictions[incorrect_predictions == 1])
    actual_negatives = len(labels[labels == 0])

    return false_positives / actual_negatives


def roc_curve(p_vals: np.array, labels: np.array, step: int=1E-5, plot: bool=True):
    thresholds = np.arange(0, 1, step)
    num_pts = len(thresholds)
    tprs = [0] * num_pts
    fprs = [0] * num_pts
    for i in range(num_pts):
        prd = predict(p_vals, thresholds[i])
        tprs[i] = tpr(prd, labels)
        fprs[i] = fpr(prd, labels)

    
    if plot:
        plt_colors = make_color_gradient((255, 128, 0), (0, 0, 255), num_pts)
        plt.scatter(fprs, tprs, c=plt_colors)
        plt.show()

    return tprs, fprs