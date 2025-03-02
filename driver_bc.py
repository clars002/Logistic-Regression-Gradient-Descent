import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from typing import List

from logistic_regression import logreg_grad_descent, probability
from evaluate import roc_curve


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
