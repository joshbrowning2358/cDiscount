import sys

import tensorflow as tf
import numpy as np

from get_bottleneck_data import get_bottleneck_data
import constants as c


graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, (c.batch_size, c.num_bottlenecks))
    targets = tf.placeholder(tf.int32, (c.batch_size))

    # FC Weights
    fc_weights = tf.Variable(tf.truncated_normal((c.num_bottlenecks, c.num_targets), stddev=0.1))
    fc_biases = tf.Variable(tf.zeros((c.num_targets)))

    # Compute the network itself:
    logits = tf.matmul(input, fc_weights) + fc_biases

    # Optimize
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=c.learning_rate).minimize(loss)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    file_name_iterator = None
    for i in range(c.num_steps):
        labels, inputs, file_iterator = get_bottleneck_data(batch_size=c.batch_size,
                                                            file_name_iterator=file_name_iterator)
        _, current_loss = session.run([optimizer, loss], feed_dict={input: inputs, targets: labels})
        sys.stdout.write('.')
        if i % 10 == 0:
            print('Loss: {}'.format(current_loss))

    weights = session.run(fc_weights)
    biases = session.run(fc_biases)

np.savetxt('weights.csv', weights, delimiter=',')
np.savetxt('biases.csv', biases, delimiter=',')
