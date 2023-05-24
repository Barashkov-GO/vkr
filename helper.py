import json
import numpy as np
import pandas as pd


def save(data, name, path):
    """
    Saving purchases data

    :param data: pandas.DataFrame
    :param name: name of file, string
    :param datasets_path: name of directory, string
    :return: None
    """
    print('SAVING dataset: {}\n'.format(name.upper()))
    data.info(memory_usage='deep')
    data.to_csv(path + name)


def get_median(x):
    x = x.values
    if len(x) == 1:
        return x[0]
    x = sorted(x)
    if len(x) % 2 == 0:
        return (x[len(x) // 2] + x[len(x) // 2 - 1]) / 2.0
    return x[(len(x) - 1) // 2]


def get_mean(x):
    if len(x) == 0:
        return None
    return np.mean(x)


def get_time_delta(x):
    x = x.values
    x = sorted(x)
    x_deltas = []
    for i in range(0, len(x) - 1):
        x_deltas.append(x[i + 1] - x[i])
    return np.array(x_deltas)


def get_quartile_range(x):
    x = pd.Series(x)
    return x.quantile(0.75) - x.quantile(0.25)


def binary_search(array, x, low, high):
    """
    Binary search algorithm
    :param array: array of numbers
    :param x: number to search
    :param low: left bound of searching
    :param high: right bound of searching
    :return: position of the x in array
    """

    while low <= high:
        mid = low + (high - low) // 2
        if array[mid] == x:
            return mid
        elif array[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return -1


class KeyDict:
    """
    The class of dictionary with needed methods to transform the data
    """

    def __init__(self, name='Default'):
        self.name = name
        self.max = 0
        self.dict = {}

    def push(self, obj):
        if obj in self.dict:
            return self.dict[obj]
        self.dict[obj] = self.max
        self.max += 1
        return self.max - 1

    def get(self, obj):
        if obj not in self.dict:
            return None
        return self.dict[obj]

    def save(self, path):
        print(len(self.dict))
        with open(path, 'w') as file:
            file.write(json.dumps(self.dict))

    def load(self, path):
        with open(path) as json_file:
            self.dict = json.load(json_file)
