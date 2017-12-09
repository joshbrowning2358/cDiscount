import sys

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import numpy as np

import constants as c


def model_fn(features, labels, mode):
    # Compute the network itself:
    logits = tf.layers.dense(inputs=features['inputs'], units=c.num_targets, activation=tf.nn.relu)

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_values = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {'classes': predicted_values, 'probabilities': probabilities}
        export_outputs = {'prediction': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate, beta1=c.beta1,
                                           beta2=c.beta2, epsilon=c.epsilon)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        label_values = tf.argmax(input=labels, axis=1)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(label_values, predicted_values)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'bottlenecks': tf.FixedLenFeature([c.num_bottlenecks], tf.float32),
            'label': tf.FixedLenFeature([c.num_targets], tf.float32),
        })
    return features['bottlenecks'], features['label']


def get_input_fn(filenames, batch_size=5):
    """
    :param filenames: List of files to read from
    :return: A function which, when called, returns the input data
    """

    def input_fn():
        filename_queue = tf.train.string_input_producer(filenames)

        image, label = read_and_decode(filename_queue)
        images, labels = tf.train.batch(
            [image, label], batch_size=batch_size,
            capacity=1000 + 3 * batch_size)

        return {'inputs': images}, labels

    return input_fn


def build_estimator(model_dir):
    est = tf.estimator.Estimator(
        model_fn=_fc_model_fn,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))
    return est


def serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32, [None, c.num_bottlenecks])}
    out = tf.estimator.export.ServingInputReceiver(inputs, inputs)
    return out
