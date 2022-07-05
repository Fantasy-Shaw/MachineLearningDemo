import pandas as pd
import numpy as np
import random


def train_test_split(src: pd.DataFrame, test_size=0.3):
    length = len(src)
    train_index = list(range(length))
    test_index = random.sample(train_index, int(length * test_size))
    for x in test_index:
        train_index.remove(x)
    return src.iloc[train_index], src.iloc[test_index]


if __name__ == "__main__":
    X = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
    print(X, "\n")
    for elem in train_test_split(X):
        print(elem, "\n")
