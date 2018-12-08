import numpy as np
import tensorflow as tf


class VGG16(object):
    def __init__(self, keep_prob, num_classes, train_layers, learning_rate=0.01, model="train", weights_path='DEFAULT'):
        """Create the graph of the VGG16 model.
       """
        # Parse input arguments into class variables
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path
        self.train_layers = train_layers

        with tf.variable_scope("input"):
            self.x_input = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x_input")
            self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        conv1_1 = conv(self.x_input, 3, 3, 64, 1, 1, name="conv1_1")
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name="conv1_2")
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, name='pool1')

        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name="conv2_1")
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name="conv2_2")
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')

        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name="conv3_1")
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name="conv3_2")
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name="conv3_3")
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')

        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name="conv4_1")
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name="conv4_2")
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name="conv4_3")
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')

        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name="conv5_1")
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name="conv5_2")
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name="conv5_3")
        pool5 = max_pool(conv5_3, 2, 2, 2, 2, name='pool5')

        fcIn = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = fc(fcIn, 512, 4096, name="fc6", relu=True)
        dropout1 = tf.nn.dropout(fc6, keep_prob)

        fc7 = fc(dropout1, 4096, 4096, name="fc7", relu=True)
        dropout2 = tf.nn.dropout(fc7, keep_prob)

        self.fc8 = fc(dropout2, 4096, num_classes, name="fc8")

        if model == "train" or model == "val":
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.y_input))

            with tf.name_scope("train"):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
                gradients = tf.gradients(self.loss, var_list)
                self.grads_and_vars = list(zip(gradients, var_list))
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars, global_step=self.global_step)

            with tf.name_scope("prediction"):
                self.prediction = tf.argmax(self.fc8, 1, name="prediction")

            with tf.name_scope("accuracy"):
                correct_prediction = tf.equal(self.prediction, tf.argmax(self.y_input, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def load_initial_weights(self, session):
        """Load weights from file into network.
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.train_layers:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(value=x,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding,
                          name=name)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(input=i, filter=k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable(name='weights',
                                  shape=[filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable(name='biases',
                                 shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable(name='weights',
                                  shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable(name='biases',
                                 shape=[num_out],
                                 trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

