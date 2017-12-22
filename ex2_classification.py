from utils import load_data, shuffle_data
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
###
from ex1_clustering import clean_labels, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
###

class MultiLayerNetwork():

    def __init__(self, data, target, batch_size=8, hidden_layer_sizes=[100, 30, 10],
                 learning_rate=0.001):
        self.learning_rate = learning_rate
        self.data = data
        self.target = target
        self.num_features = np.shape(self.data)[1]
        self.num_classes = np.shape(self.target)[1]
        self.data_ph = tf.placeholder(tf.float32, [None, self.num_features])
        self.target_ph = tf.placeholder(tf.float32, [None, self.num_classes])
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.output = None
        self.training = None
        self.evaluation = None

    def build_inference(self):
        if self.output == None:
            input_layer_weights = tf.Variable(tf.truncated_normal([self.num_features, self.hidden_layer_sizes[0]]))
            input_layer_bias = tf.Variable(tf.constant(.1, shape=[self.hidden_layer_sizes[0]]))
            input_layer_output = tf.matmul(self.data_ph, input_layer_weights) + input_layer_bias
            previous_layer_output = tf.sigmoid(input_layer_output)
            for i in range(len(self.hidden_layer_sizes) - 1):
                temp_weights = tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]]))
                temp_biases = tf.Variable(tf.constant(.1, shape=[self.hidden_layer_sizes[i+1]]))
                hidden_layer_output = tf.sigmoid(tf.matmul(previous_layer_output, temp_weights) + temp_biases)
                previous_layer_output = hidden_layer_output
            output_layer_weights = tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[-1], self.num_classes]))
            output_layer_biases = tf.Variable(tf.constant(.1, shape=[self.num_classes]))
            self.output = tf.matmul(previous_layer_output, output_layer_weights) + output_layer_biases

        return self.output

    def build_loss_function(self):
        activation = tf.sigmoid(self.output)
        pos = self.target_ph * tf.log( activation )
        neg = (1.0 - self.target_ph) * tf.log( 1.0 - activation )
        per_example_error = tf.reduce_sum(pos + neg,1)
        result = -tf.reduce_mean( per_example_error )

        self.loss_function = result

    def build_train(self):
        if self.training == None:
            self.build_loss_function()
            #self.loss_function = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_ph, logits=self.output)
            self.training = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)
        return self.training

    def build_evaluate(self):
        if self.evaluation == None:
            self.evaluation = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.target_ph, 1), tf.argmax(self.output, 1)), tf.float32))
        return self.evaluation

    def train_epoch(self, X_test, y_test, num_epochs=1):
        if self.output == None:
            self.build_inference()
            self.build_train()
            self.build_evaluate()
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
        num_batches = int(len(self.data) / self.batch_size)
        for i in range(num_epochs):
            data_train, target_train = shuffle_data(self.data, self.target)
            for j in range(num_batches):
                batch_data = data_train[j * self.batch_size:(j+1) * self.batch_size]
                batch_target = target_train[j * self.batch_size:(j+1) * self.batch_size]
                _, loss = self.session.run([self.training, self.loss_function],
                                 {self.data_ph: batch_data, self.target_ph: batch_target})
            accuracy, loss, out, target = self.session.run([self.evaluation, self.loss_function,
                                                            self.output, self.target_ph], {self.data_ph: X_test, self.target_ph: y_test})
            print('Epoch {}; Loss {}; accuracy {}'.format(i, loss, accuracy))

if __name__ == '__main__':
    tf.reset_default_graph()
    all_data = load_data('breast-cancer-wisconsin.data')
    #data, target = load_iris(True)

    # remember to one-hot encode your data:
    # target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))

    ###
    data = all_data[:, 1:-1]
    target = all_data[:, -1]
    data = preprocess_data(data)
    target = clean_labels(target)
    target = OneHotEncoder(sparse=False).fit_transform(target.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=33)
    multilayer = MultiLayerNetwork(X_train, y_train)
    multilayer.train_epoch(X_test, y_test, num_epochs=1000)
    ###
