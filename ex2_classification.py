from utils import load_data, shuffle_data, MultiLayerNetwork
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
###
from ex1_clustering import clean_labels, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
###


class MyNetwork(MultiLayerNetwork):

    def __init__(self, data, target):
        super().__init__(data, target)

    def build_train(self):
        # your work here...

    def build_evaluate(self):
        super().build_evaluate()

    def build_train(self):
        super().build_train()

    def train_epoch(self, X_test, y_test, num_epochs=1):
        super().train_epoch(X_test, y_test, num_epochs=num_epochs)

if __name__ == '__main__':
    all_data = load_data('breast-cancer-wisconsin.data')
    # remember to one-hot encode your data:
    # target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))

    # your work here...
