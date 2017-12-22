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

    def build_loss(self):
        # your work here...
        activation = tf.sigmoid(self.output)
        pos = self.target_ph * tf.log( activation )
        neg = (1.0 - self.target_ph) * tf.log( 1.0 - activation )
        per_example_error = tf.reduce_sum(pos + neg,1)
        result = -tf.reduce_mean( per_example_error )
        return result

if __name__ == '__main__':
    tf.reset_default_graph()
    all_data = load_data('breast-cancer-wisconsin.data')
    # remember to one-hot encode your data:
    # target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))

    ###
    # your work here...
    data = all_data[:, 1:-1]
    target = all_data[:, -1]
    data = preprocess_data(data)
    target = clean_labels(target)
    target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=33)
    multilayer = MyNetwork(X_train, y_train)
    multilayer.train(X_test, y_test, num_epochs=300)
    ###
