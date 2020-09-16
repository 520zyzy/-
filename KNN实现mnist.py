from keras.datasets import mnist
import numpy as np
import pandas as pd

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = np.where(train_data > 0, 1, 0)
test_data = np.where(test_data > 0, 1, 0)


def countXsd(train_data, test_data):
    n = 0
    for test, testlab in zip(test_data[:1000], test_labels[:1000]):
        n = n + fenlei(train_labels, np.sum(np.sum(((train_data - test) * (train_data - test)), axis=1), axis=1),
                       testlab)
    return n


def fenlei(train_labels, test_train, testlab):
    t_array = np.vstack((test_train, train_labels))
    t_array = t_array[:, t_array[0].argsort()]
    # 排序
    d = dict(pd.DataFrame(t_array[:, :100]).loc[1].value_counts())
    n = max(d, key=d.get)
    if n == testlab:
        return 1
    else:
        return 0


print(countXsd(train_data, test_data) / test_data.shape[0])
