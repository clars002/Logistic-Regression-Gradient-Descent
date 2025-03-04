# This file is the main driver to apply my classifier to the breast cancer dataset (HW3 Problem 4)
import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

from evaluate import confusion_matrix, roc_curve
from logistic_regression import logreg_grad_descent, predict, probability


def process_args():
    """
    Parses arguments from the CLI.

    Returns:
        A Namespace object where attributes correspond to the
        defined/provided args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=131_072,
        help="Number of iterations of gradient descent to perform.",
    )
    parser.add_argument(
        "--learnrate",
        type=float,
        default=1e-8,
        help="Learning rate (i.e. step size for gradient descent) Note: Larger than 1E-8 can cause overflow.",
    )
    parser.add_argument(
        "--showloss",
        action="store_true",
        help="Display the curve of the loss function over the iterations of gradient descent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=64,
        help="Seed for randomization (used when sampling the data).",
    )
    parser.add_argument(
        "--confmatrix",
        action="store_true",
        help="Display the confusion matrix based on the test data.",
    )
    return parser.parse_args()


# Read in the input and use random sampling to split into training and test sets
def process_input(rand_seed: int = 64):
    dataset = skl.datasets.load_breast_cancer()

    total_size = len(dataset["data"])
    training_size = int((total_size / 3) * 2)
    test_size = total_size - training_size

    np.random.seed(rand_seed)
    training_indices = np.random.choice(
        range(total_size), size=training_size, replace=False
    )
    test_indices = [_ for _ in range(total_size) if _ not in training_indices]

    training_data = dataset["data"][training_indices]
    training_labels = dataset["target"][training_indices]

    test_data = dataset["data"][test_indices]
    test_labels = dataset["target"][test_indices]

    return training_data, training_labels, test_data, test_labels


# Display initial/final loss and plot loss vs iterations
def display_loss_stats(losses, iterations):
    print(f"Initial loss: {losses[0]}")
    print(f"Final loss: {losses[iterations - 1]}\n")

    plt.figure(2)

    plt.plot(range(iterations), losses)
    plt.plot(range(iterations), losses, marker="o", alpha=0.5)

    plt.title("Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    return


# Perform the gradient descent and generate/display the ROC curve
if __name__ == "__main__":
    args = process_args()
    seed = args.seed
    iterations = args.iterations
    learn_rate = args.learnrate
    show_loss = args.showloss
    show_conf_matrix = args.confmatrix

    tr_data, tr_labels, tst_data, tst_labels = process_input(seed)
    print(
        f"Finished reading in dataset. Training and test subsets generated with seed: {seed}.\n"
    )

    # Start at all weights equal to zero
    w_0 = np.zeros((30,))

    print(
        f"Executing gradient descent ({iterations} iterations, step of {learn_rate}). Please wait..."
    )
    solution, losses = logreg_grad_descent(
        w_0, tr_data, tr_labels, learn_rate, iterations
    )
    print("Finished.\n")

    tst_prob = probability(tst_data, solution)

    roc_curve(tst_prob, tst_labels)

    if show_loss:
        display_loss_stats(losses, iterations)

    if show_conf_matrix:
        tst_predictions = predict(tst_prob, 0.5)
        cm = confusion_matrix(tst_predictions, tst_labels)
        print(f"Confusion Matrix (threshold = 0.5):\n{cm}\n")

    plt.show()
