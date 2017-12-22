import pandas
import numpy as np
import tensorflow as tf
import numpy as np

def load_data(path):
    dataframe = pandas.read_csv(path, header=None)
    temp = []
    for d in dataframe.values:
        if '?' in d:
            np.place(d, d=='?', np.nan)
        temp.append(d)
    return np.array(temp)

def shuffle_data(a, b):
    assert np.shape(a)[0] == np.shape(b)[0]
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

    def build_loss(self):
        return tf.nn.softmax_cross_entropy_with_logits(labels=self.target_ph, logits=self.output)

    def build_train(self):
        if self.training == None:
            self.loss_function = self.build_loss()
            self.training = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)
        return self.training

    def build_evaluate(self):
        if self.evaluation == None:
            self.evaluation = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.target_ph, 1), tf.argmax(self.output, 1)), tf.float32))
        return self.evaluation

    def train(self, X_test, y_test, num_epochs=1):
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
                self.session.run([self.training, self.loss_function],
                                 {self.data_ph: batch_data, self.target_ph: batch_target})
            accuracy, loss = self.session.run([self.evaluation, self.loss_function], {self.data_ph: X_test, self.target_ph: y_test})
            print('Epoch {}; Loss {}; accuracy {}'.format(i, loss, accuracy))



if __name__ == '__main__':
    data = load_data('breast-cancer-wisconsin.data')
    print(data)
