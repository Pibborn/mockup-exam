from utils import load_data, shuffle_data, MultiLayerNetwork
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder



class MyNetwork(MultiLayerNetwork):

    def __init__(self, data, target):
        super().__init__(data, target)

    def build_loss(self):
        # your work here...
        #
        # You can assume:
        #   - self.output is a tensor representing the output units.
        #       It has shape (?, 2).
        #   - self.target_ph is a tensor representing the target balues.
        #       It has shape (?, 2).
        #
        # Example:
        return tf.reduce_mean(tf.norm(self.output - self.target_ph, axis=1))


if __name__ == '__main__':
    all_data = load_data('breast-cancer-wisconsin.data')

    # remember to one-hot encode your data:
    # target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))

    # your work here...
    # ...
