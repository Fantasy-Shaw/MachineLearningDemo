import pandas as pd
import numpy as np
import random


def bootstrapping(src: pd.DataFrame):
    length = len(src)
    trainIndex = []
    testIndex = list(range(length))
    for i in range(length):
        tmp = random.randint(0, length - 1)
        trainIndex.append(tmp)
        if tmp in testIndex:
            testIndex.remove(tmp)
    # 去重
    trainIndex = list(set(trainIndex))
    return src.iloc[trainIndex], src.iloc[testIndex]


if __name__ == "__main__":
    X = pd.DataFrame(np.random.randn(15, 4), columns=['A', 'B', 'C', 'D'])
    print(X, "\n")
    for elem in bootstrapping(X):
        print(elem, "\n")
