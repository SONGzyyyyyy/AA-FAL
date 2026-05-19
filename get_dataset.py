import numpy as np
import random


def PreTrainDataset_prepared(ft, classi):
    x, y = WiFi_Dataset_slice(ft, classi)
    train_index_shot = []
    for i in classi:
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:1000]
    X_train = x[train_index_shot]
    Y_train = y[train_index_shot].astype(np.uint8)

    max_value = X_train.max()
    min_value = X_train.min()
    X_train = (X_train - min_value) / (max_value - min_value)
    return X_train, Y_train

def FineTuneDataset_prepared(ft, classi, k, seed):
    x, y = WiFi_Dataset_slice(ft, classi)
    test_index_shot = []
    finetune_index_shot = []
    for i in classi:
        i -= classi[0]
        index_classi = [index for index, value in enumerate(y) if value == i]
        random.seed(seed)
        finetune_index_shot += random.sample(index_classi[0:1000], k)
        test_index_shot += index_classi[1000:2000]
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]

    max_value = X_train.max()
    min_value = X_train.min()
    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)
    return X_train, X_test, Y_train, Y_test
