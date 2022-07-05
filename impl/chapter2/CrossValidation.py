import pandas as pd
import numpy as np


def cross_validation(src: pd.DataFrame):
    splits = []
    length = len(src)
    train_index = list(range(length))
    for _thisTestIndex in range(length):
        _thisTrainIndex = train_index.copy()
        _thisTrainIndex.remove(_thisTestIndex)
        splits.append([src.iloc[_thisTrainIndex], src.iloc[_thisTestIndex]])
    return splits


if __name__ == "__main__":
    X = pd.DataFrame(np.random.randn(5, 4), columns=['A', 'B', 'C', 'D'])
    print(X, "\n")
    for split in cross_validation(X):
        for elem in split:
            print(elem, "\n")
