import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

import model


def generate_experiment_fn(data_dir,
                           train_batch_size=100,
                           eval_batch_size=100,
                           train_steps=10000,
                           eval_steps=100,
                           **experiment_args):

    def _experiment_fn(output_dir):
        return Experiment(
            tf.estimator.Estimator(model_fn=model.model_fn, model_dir=output_dir,
                                   config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180)),
            train_input_fn=model.get_input_fn(data_dir + '/train.tfrecords', batch_size=train_batch_size),
            eval_input_fn=model.get_input_fn(data_dir + '/eval.tfrecords', batch_size=eval_batch_size),
            export_strategies=[saved_model_export_utils.make_export_strategy(
                model.serving_input_fn,
                default_output_alternative_key=None,
                exports_to_keep=1)],
            train_steps=train_steps,
            eval_steps=eval_steps,
            **experiment_args
        )

    return _experiment_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='/Users/joshuabrowning/Personal/Kaggle/cDiscount/tf_files/bottlenecks_small_example/',
        help='GCS or local path to training data',
    )
    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=100
    )
    parser.add_argument(
        '--eval_batch_size',
        help='Batch size for evaluation steps',
        type=int,
        default=100
    )
    parser.add_argument(
        '--train_steps',
        help='Steps to run the training job for.',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--eval_steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int
    )
    parser.add_argument(
        '--output_dir',
        default='./temp',
        help='GCS location to write checkpoints and export models',
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help='Minimum number of training steps between evaluations',
        default=1,
        type=int
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    output_dir = arguments.pop('output_dir')

    # Run the training job
    print('-'*120)
    print('Running job!')
    print('-'*120)
    learn_runner.run(generate_experiment_fn(**arguments), output_dir)
