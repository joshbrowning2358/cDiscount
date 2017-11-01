import sys

import tensorflow as tf
import numpy as np

from get_bottleneck_data import get_bottleneck_data
import constants as c


class BottleneckReader(object):
    """
    This class is just for keeping track of the current state of the data reading
    """
    def __init__(main, current_features=None, current_target=None, current_chunk_id=0):
        self.current_features = current_features
        self.current_target = current_target
        self.current_chunk_id = current_chunk_id

    def get_data(data_dir, batch_size):
        features, labels, self.current_features, self.current_target, self.current_chunk_id =\
            get_bottleneck_data(data_dir, batch_size, self.current_features, self.current_target,
                                self.current_chunk_id)


reader = BottleneckReader()


def _fc_model_fn(features, labels, mode):
    # Compute the network itself:
    logits = tf.dense(inputs=features, units=c.num_targets, activation=tf.nn.relu)

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=c.num_targets), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)


def input_fn(file_path, batch_size, train):
    if train:
        reader.get_data(file_path, batch_size)
    else:
        reader.get_data(file_path + '/eval', batch_size)


def get_input_fn(file_path, batch_size, train):
    return lambda: input_fn(file_path, batch_size, train)


def build_estimator(model_dir):
  return tf.estimator.Estimator(
      model_fn=_fc_model_fn,
      model_dir=model_dir,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
  inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

