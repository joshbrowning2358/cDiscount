import os

import numpy as np

import constants as c

label_cnts = np.genfromtxt('output/distinct_categories.csv', delimiter=',', dtype=[int, int])
TOP_CLASSES = [k for k, v in label_cnts if v > 500] # Over 1,600 categories, and not all data yet
TOP_CLASSES.sort()

# DATA_DIR = '/Users/joshuabrowning/Personal/Kaggle/cDiscount/tf_files/bottlenecks'
DATA_DIR = '/Users/joshuabrowning/Desktop/test_tf_files'


def get_bottleneck_data(batch_size=64, current_features=None, current_target=None,
					    current_chunk_id=0, classes=TOP_CLASSES):
    """
    Returns labels and features for batch_size observations.  Also returns current_features and current_target with 
    data removed, as well as the new current_chunk_id.  If current_data is None, current_chunk_id will be used to read
    the next dataset and then the value incremented.
    :param batch_size: How many observations should be returned?
    :param current_features: Numpy array of the features.
    :param current_target: Numpy array of the target.
    :param current_chunk_id: Number indicating which chunk to read next.
    :param classes: list of the classes to map class to id
    :return: A tuple of labels, image features, and updated inputs for next round.
    """
    if current_features is not None and current_features.shape[0] < batch_size:
    	current_features = None
    
    if current_features is None:
    	current_features = np.genfromtxt(DATA_DIR + '/chunked_file_{}.txt'.format(current_chunk_id))
    	current_target = np.genfromtxt(DATA_DIR + '/chunked_labels_{}.txt'.format(current_chunk_id))
    	# Shuffle data randomly
    	perm = np.random.permutation(current_features.shape[0])
    	current_features = current_features[perm, :]
    	current_target = current_target[perm]

    	current_chunk_id += 1
    	if current_chunk_id == 10:
    		current_chunk_id = 0
    
    bottlenecks = current_features[:batch_size]
    labels = [get_class_id(x, classes) for x in current_target[:batch_size]]

    return bottlenecks, labels, current_features[batch_size:], current_target[batch_size:], current_chunk_id


def get_class_id(category_id, classes):
    index = [i for i, x in enumerate(classes) if int(category_id) == x]
    if len(index) == 0:
        result = len(classes)
    else:
        result = index[0]
    return result

if __name__ == '__main__':
    bs = 256
    from time import time

    start = time()
    labs, data, c_feat, c_target, c_chunk_id = get_bottleneck_data(batch_size=bs, current_chunk_id=1)
    print('First run takes {}s'.format(round(time() - start, 3)))
    print('First three labels: {}'.format(labs[:3]))
    print('First 3x3 features: {}'.format(data[:3, :3]))
    print('Remaining features: {}'.format(c_feat.shape[0]))
    print('Remaining labels: {}'.format(c_target.shape[0]))

    start = time()
    labs, data, c_feat, c_target, c_chunk_id = get_bottleneck_data(bs, c_feat, c_target, c_chunk_id)
    print('Second run takes {}s'.format(round(time() - start, 3)))

    start = time()
    labs, data, c_feat, c_target, c_chunk_id = get_bottleneck_data(bs, c_feat, c_target, c_chunk_id)
    print('Third run takes {}s'.format(round(time() - start, 3)))

    start = time()
    labs, data, c_feat, c_target, c_chunk_id = get_bottleneck_data(bs, c_feat, c_target, c_chunk_id)
    print('Fourth run takes {}s'.format(round(time() - start, 3)))

