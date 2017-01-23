import tensorflow as tf
import numpy as np


class QFuncModel():
    def __init__(self, args):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # input layer
        self.s = tf.placeholder("float", [None, args.resize_width, args.resize_height, args.frames])
        self.a = tf.placeholder("float", [None, args.actions])
        self.y = tf.placeholder("float", [None])

        self.W_conv1 = weight_variable([8, 8, 4, 32])
        self.b_conv1 = bias_variable([32])

        self.W_conv2 = weight_variable([4, 4, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.W_conv3 = weight_variable([3, 3, 64, 64])
        self.b_conv3 = bias_variable([64])

        self.W_fc1 = weight_variable([1600, 512])
        self.b_fc1 = bias_variable([512])

        self.W_fc2 = weight_variable([512, args.actions])
        self.b_fc2 = bias_variable([args.actions])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2, 2) + self.b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)

        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

        # readout layer
        self.readout = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

        # define the cost function
        readout_action = tf.reduce_sum(tf.mul(self.readout, self.a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.y - readout_action))
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    def variable_list(self):
        return [self.W_conv1, self.W_conv2, self.W_conv3, self.W_fc1, self.W_fc2,
                self.b_conv1, self.b_conv2, self.b_conv3, self.b_fc1, self.b_fc2 ]

    def copy(self, sess, model2):
        l1 = self.variable_list()
        l2 = model2.variable_list()
        assign_op = [tf.assign(l1[i], l2[i]) for i in range(len(l1))]
        sess.run(assign_op)