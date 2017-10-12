import pandas as pd
import numpy as np

import tensorflow as tf

from get_train_data import get_train_data
import constants as c


graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, (c.batch_size, c.width, c.height, c.alpha_channels))
    targets = tf.placeholder(tf.int32, (c.batch_size))

    # Convolutional Weights
    conv_w = tf.Variable(tf.truncated_normal((c.filter_height, c.filter_width, c.alpha_channels,
                                              c.convolutional_channels), stddev=0.1))
    conv_b = tf.Variable(tf.zeros((c.convolutional_channels)))

    # FC Weights
    fc_input_node_cnt = int(c.convolutional_channels * c.width * c.height /
                            c.convolutional_skip ** 2 / c.pool_skip ** 2)
    fc_weights = tf.Variable(tf.truncated_normal((fc_input_node_cnt, c.num_targets), stddev=0.1))
    fc_biases = tf.Variable(tf.zeros((c.num_targets)))

    # Compute the network itself:
    conv_output = tf.nn.conv2d(input, conv_w, [1, c.convolutional_skip, c.convolutional_skip, 1], 'SAME')
    pool_output = tf.nn.max_pool(conv_output, [1, c.pool_skip, c.pool_skip, 1], [1, c.pool_skip, c.pool_skip, 1],
                                 padding='SAME')
    pool_output = tf.nn.relu(pool_output + conv_b)
    fc_input = tf.reshape(pool_output, (c.batch_size, -1))
    logits = tf.matmul(fc_input, fc_weights) + fc_biases

    # Optimize
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=c.learning_rate).minimize(loss)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    data_generator = None
    for i in range(c.num_steps):
        labels, images, data_generator = get_train_data(batch_size=batch_size, data_generator=data_generator,
                                                        image_resize=width)
        _, current_loss = session.run([optimizer, loss], feed_dict={input: images, targets: labels})
        print('Loss: {}'.format(current_loss))