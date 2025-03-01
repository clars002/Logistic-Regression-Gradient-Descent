import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from typing import List


def logreg_loss(X: np.array, y:np.array, w: np.array):
    Xw = np.matmul(X, w)
    return np.sum(np.log(1 + np.exp(Xw))) - np.matmul(y.transpose(), Xw)


def probability(X, w):
    Xw = np.matmul(X, w)
    p = np.exp(Xw)
    p = p / (1 + p)

    return p


def logreg_gradient(X: np.array, y: np.array, w: np.array):
    p = probability(X, w)

    return np.matmul(X.transpose(), (p - y))


def logreg_grad_descent(w_0: np.array, X: np.array, y: np.array, step: float, iterations: int):
    w_k = w_0
    losses = [0] * iterations
    for i in range(iterations):
        search_dir = -logreg_gradient(X, y, w_k)
        w_k = w_k + (step * search_dir)
        cur_loss = logreg_loss(X, y, w_k)
        losses[i] = cur_loss
    return w_k, losses


def predictions(p_vals: np.array, threshold: float):
    predictions = p_vals.copy()
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0

    return predictions


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


def interpolate(start: int, end: int, num_pts: int):
    diff = end - start
    step = diff / (num_pts)
    print(diff)
    print(step) 
    print(np.arange(start, end, step))
    return np.arange(start, end, step)

def roc_curve(p_vals: np.array, labels: np.array, step: int=1E-5, plot: bool=True):
    thresholds = np.arange(0, 1, step)
    num_pts = len(thresholds)
    tprs = [0] * num_pts
    fprs = [0] * num_pts
    for i in range(num_pts):
        prd = predictions(p_vals, thresholds[i])
        tprs[i] = tpr(prd, labels)
        fprs[i] = fpr(prd, labels)

    color_gradient = np.array([interpolate(255, 0, num_pts), interpolate(165, 0, num_pts), interpolate(0, 255, num_pts)])
    colors = [0] * num_pts
    for i in range(num_pts):
        colors[i] = (float(color_gradient[0][i] / 255), float(color_gradient[1][i] / 255), float(color_gradient[2][i] / 255))
    
    if plot:
        plt.scatter(fprs, tprs, c=colors)
        plt.show()

    return tprs, fprs
    

def process_input(rand_seed: int=64):
    dataset = skl.datasets.load_breast_cancer()

    total_size = len(dataset['data'])
    training_size = int((total_size / 3) * 2)
    test_size = total_size - training_size

    np.random.seed(rand_seed)
    training_indices = np.random.choice(range(total_size), size=training_size, replace=False)
    test_indices = [_ for _ in range(total_size) if _ not in training_indices]

    training_data = dataset['data'][training_indices]
    training_labels = dataset['target'][training_indices]

    test_data = dataset['data'][test_indices]
    test_labels = dataset['target'][test_indices]
    
    return training_data, training_labels, test_data, test_labels


if __name__ == "__main__":
    iterations = 131_072 
    learn_rate = 1E-8
    tr_data, tr_labels, tst_data, tst_labels = process_input()

    w_0 = np.zeros((30,))

    solution, losses = logreg_grad_descent(w_0, tr_data, tr_labels, learn_rate, iterations)

    print(f"initial loss: {losses[0]}")
    print(f"final loss: {losses[iterations - 1]}")
    test = probability(tst_data, solution)

    roc_curve(test, tst_labels)
