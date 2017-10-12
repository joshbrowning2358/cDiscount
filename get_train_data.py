import io

import bson
from skimage.data import imread
from skimage.transform import resize
import numpy as np
import _pickle as pickle


IMG_CHANNELS = 3

# with open('distinct_categories.p', 'rb') as file_:
#     label_cnts = pickle.load(file=file_)
# TOP_CLASSES = [k for k, v in label_cnts.items() if v > 1000]
# TOP_CLASSES.sort()

with open('distinct_categories_example.p', 'rb') as file_:
    label_cnts = pickle.load(file=file_)
TOP_CLASSES = [[k for k, v in label_cnts.items() if v > 35][0]]
TOP_CLASSES.sort()


def get_train_data(batch_size=10, data_generator=None, data_file='data/train_example.bson', classes=TOP_CLASSES,
                   image_resize=180):
    """
    Returns a batch_size of data as well the updated data_generator.
    :param batch_size: How many observations should be returned?
    :param data_generator: The current data generator.  If None, it will start with the file.
    :param data_file: Where should the data be pulled from?
    :return: A tuple of labels (or nan's if data is the test set), image data, and the reduced data_generator.
    """
    images = np.zeros((batch_size, image_resize, image_resize, IMG_CHANNELS))
    labels = []

    index = 0
    while index < batch_size:
        if data_generator is None:
            data_generator = bson.decode_file_iter(open(data_file, 'rb'))

        try:
            data_dict = data_generator.__next__()
        except StopIteration:
            data_generator = bson.decode_file_iter(open(data_file, 'rb'))
            data_dict = data_generator.__next__()

        # product_id = data_dict['_id']
        category_id = data_dict.get('category_id', np.nan)
        label = get_class_id(category_id, classes)

        images_to_append = min(batch_size - index, len(data_dict['imgs']))
        for e, pic in enumerate(data_dict['imgs'][:images_to_append]):
            picture = imread(io.BytesIO(pic['picture']))
            picture = resize(picture, (image_resize, image_resize, IMG_CHANNELS), mode='constant')
            images[index, :, :, :] = picture
            labels += [label]
            index += 1

    return labels, images, data_generator


def get_class_id(category_id, classes):
    index = [i for i, x in enumerate(classes) if category_id == x]
    if len(index) == 0:
        result = len(classes)
    else:
        result = index[0]
    return result

# labels = []
# data_generator = bson.decode_file_iter(open('data/train.bson', 'rb'))
# for d in data_generator:
#     labels += [d['category_id']]
# with open('train_labels.p', 'wb') as file_:
#     pickle.dump(labels, file=file_)
#
# from itertools import groupby
# labels.sort()
# label_cnts = {k: len(list(v)) for k, v in groupby(labels)}
# with open('distinct_categories.p', 'wb') as file_:
#     pickle.dump(label_cnts, file=file_)
# with open('distinct_categories.csv', 'w') as f:
#     for k, v in label_cnts.items():
#         f.writelines(str(k) + ',' + str(v) + '\n')


# labels = []
# data_generator = bson.decode_file_iter(open('data/train_example.bson', 'rb'))
# index = 0
# for d in data_generator:
#     labels += [d['category_id']]
# with open('train_example_labels.p', 'wb') as file_:
#     pickle.dump(labels, file=file_)
#
# from itertools import groupby
# labels.sort()
# label_cnts = {k: len(list(v)) for k, v in groupby(labels)}
# with open('distinct_categories_example.p', 'wb') as file_:
#     pickle.dump(label_cnts, file=file_)
