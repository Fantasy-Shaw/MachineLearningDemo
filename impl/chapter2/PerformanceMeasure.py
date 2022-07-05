import pandas as pd
import numpy as np
import random


def mean_squared_error(predicts: list, realities: list):
    if len(predicts) != len(realities):
        return
    m = len(predicts)
    err_sum = 0.0
    for i in range(m):
        err_sum += (predicts[i] - realities[i]) ** 2
    return err_sum / m


def mean_squared_error_2(x_begin, x_end, predict, probability_density, reality, x_nums=100):
    """积分形式，参数依次为：积分下限、积分上限、预测结果f(x)、概率密度p(x)、真实标记、采样个数"""
    x_set = np.linspace(x_begin, x_end, x_nums)
    _credit = 0.0
    for x in x_set:
        y = (predict(x) - reality(x)) ** 2 * probability_density(x)
        _credit += y * (x_end - x_begin) / x_nums
    return _credit


def err(predicts: list, realities: list):
    if len(predicts) != len(realities):
        return
    m = len(predicts)
    err_sum = 0.0
    for i in range(m):
        err_sum += int(predicts[i] != realities[i])
    return err_sum / m


def acc(predicts: list, realities: list):
    return 1.0 - err(predicts, realities)


def classifyResultMatrix(predicts: list, realities: list, positive=1):
    """tp, fp, fn, tn"""
    if len(predicts) != len(realities):
        return
    m = len(predicts)
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(m):
        if predicts[i] == positive:
            if realities[i] == positive:
                tp += 1  # true positive
            else:
                fp += 1
        else:
            if realities[i] == positive:
                fn += 1  # false negative
            else:
                tn += 1
    return [tp, fp, fn, tn]


def precision(predicts: list, realities: list, positive=1):
    tp, fp, fn, tn = classifyResultMatrix(predicts, realities, positive)
    return tp / (tp + fp)


def recall(predicts: list, realities: list, positive=1):
    tp, fp, fn, tn = classifyResultMatrix(predicts, realities, positive)
    return tp / (tp + fn)


def F_beta(predicts: list, realities: list, beta: float, positive=1):
    _P = precision(predicts, realities, positive)
    _R = recall(predicts, realities, positive)
    return (1 + beta ** 2) * _P * _R / ((beta ** 2 * _P) + _R)


def F_1(predicts: list, realities: list, positive=1):
    return F_beta(predicts, realities, 1.0, positive)
