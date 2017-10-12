import os

import numpy as np

import constants as c

label_cnts = np.genfromtxt('distinct_categories.csv', delimiter=',', dtype=[int, int])
TOP_CLASSES = [k for k, v in label_cnts if v > 500] # Over 1,600 categories, and not all data yet
TOP_CLASSES.sort()

DATA_DIR = '/Users/joshuabrowning/Personal/Kaggle/cDiscount/tf_files/bottlenecks'


def get_bottleneck_data(batch_size=64, file_name_iterator=None, classes=TOP_CLASSES):
    """
    Returns a batch_size of data as well the updated data_iterator.
    :param batch_size: How many observations should be returned?
    :param file_name_iterator: The current data iterator.  If None, it will recreate the iterator.
    :param classes: list of the classes to map class to id
    :return: A tuple of labels, image data, and the reduced data_iterator.
    """
    bottlenecks = np.zeros((batch_size, c.num_bottlenecks))
    labels = []

    for index in range(batch_size):
        if file_name_iterator is None:
            file_name_iterator = get_file_name_iterator()

        try:
            data = file_name_iterator.next()
        except StopIteration:
            file_name_iterator = get_file_name_iterator()
            data = file_name_iterator.next()

        category_id = data[0]
        labels += [get_class_id(category_id, classes)]
        bottlenecks[index, :] = np.genfromtxt(DATA_DIR + '/' + category_id + '/' + data[1], delimiter=',')

    return labels, bottlenecks, file_name_iterator


def get_file_name_iterator():
    data_files = []
    categories = os.listdir(DATA_DIR)
    for category in categories:
        data_files += [(category, x) for x in os.listdir(DATA_DIR + '/' + category)]
    np.random.shuffle(data_files)
    return iter(data_files)


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
    labs, data, it = get_bottleneck_data(batch_size=bs)
    print('First run takes {}s'.format(round(time() - start, 3)))

    start = time()
    labs, data, it = get_bottleneck_data(batch_size=bs, file_name_iterator=it)
    print('Second run takes {}s'.format(round(time() - start, 3)))

    start = time()
    labs, data, it = get_bottleneck_data(batch_size=bs, file_name_iterator=it)
    print('Third run takes {}s'.format(round(time() - start, 3)))

    start = time()
    labs, data, it = get_bottleneck_data(batch_size=bs, file_name_iterator=it)
    print('Fourth run takes {}s'.format(round(time() - start, 3)))