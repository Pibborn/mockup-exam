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
<<<<<<< Updated upstream
=======
    data = all_data[:, 1:-1]
    target = all_data[:, -1]
    data = preprocess_data(data)
    target = clean_labels(target)
    target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=33)
    multilayer = MyNetwork(X_train, y_train)
    multilayer.train_epoch(X_test, y_test, num_epochs=200)
    ###
>>>>>>> Stashed changes
