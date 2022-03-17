from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def preProcess(n):
    db = load_iris()
    X_total = db['data']
    y_total = db['target']
    c0 = 0  # setosa
    c1 = 1  # versicolor
    sub_index = np.logical_or(y_total == c0, y_total == c1)
    X = X_total[sub_index]
    y = y_total[sub_index]
    y[y == c0] = -1  # setosa
    y[y == c1] = 1  # versicolor
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    return X_train[:, :n], X_test[:, :n], y_train, y_test
