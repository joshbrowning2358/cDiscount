import os
import re
import shutil
import fileinput

import numpy as np
import tensorflow as tf


chunk_location = '/Users/joshuabrowning/Personal/Kaggle/cDiscount/tf_files/bottlenecks_v2'

current_dirs = os.listdir(chunk_location)
current_dirs = [x for x in current_dirs if not re.match('.*(txt|tfrecords)', x)]

all_classes = np.genfromtxt('/Users/joshuabrowning/Personal/Kaggle/cDiscount/output/distinct_categories.csv',
                            delimiter=',')
classes = np.array([id for id, cnt in all_classes if cnt > 500])
classes.sort()
writer = tf.python_io.TFRecordWriter(chunk_location + '/train.tfrecords')

index = 0
for dir in current_dirs:
    base = '{}/{}'.format(chunk_location, dir)
    filenames = [base + '/' + x for x in os.listdir(base) if re.match('.*mobilenet.*', x)]
    f_in = fileinput.input(filenames)
    for line in f_in:
        label = np.zeros(len(classes) + 1)
        class_index = [i for i, x in enumerate(classes) if x == float(dir)]
        if len(class_index) == 0:
            class_index = len(classes)
        else:
            class_index = class_index[0]
        label[class_index] = 1
        feature_array = np.array([float(x) for x in line.split(',')])
        feature = {'label': tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                   'bottlenecks': tf.train.Feature(float_list=tf.train.FloatList(value=feature_array))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    f_in.close()
    index += 1
    print('Processed directory {} ({}%)'.format(dir, round(float(index)/len(current_dirs)*100, 2)))
