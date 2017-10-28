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
    fc_weights = tf.Variable(tf.truncated_normal((c.num_bottlenecks, c.num_targets), stddev=0.01))
    fc_biases = tf.Variable(tf.zeros((c.num_targets)))

    # Compute the network itself:
    logits = tf.matmul(input, fc_weights) + fc_biases

    # Optimize
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=c.learning_rate).minimize(loss)


losses = []

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    remaining_inputs, remaining_labels = None, None
    chunk_id = 1
    for i in range(c.num_steps):
        inputs, labels, remaining_inputs, remaining_labels, chunk_id = get_bottleneck_data(c.batch_size, remaining_inputs, remaining_labels, chunk_id)
        _, current_loss = session.run([optimizer, loss], feed_dict={input: inputs, targets: labels})
        sys.stdout.write('.')
        losses += [current_loss]
        if i % 25 == 24:
            print('Average loss: {}'.format(np.mean(losses[len(losses) - 24:])))
            np.savetxt('output/losses.csv', np.array(losses), delimiter=',')

    weights = session.run(fc_weights)
    biases = session.run(fc_biases)

np.savetxt('output/weights.csv', weights, delimiter=',')
np.savetxt('output/biases.csv', biases, delimiter=',')
np.savetxt('output/losses.csv', np.array(losses), delimiter=',')

