import io
import os
import time

import bson
from skimage.data import imread
from scipy.misc import toimage
import numpy as np


IMG_CHANNELS = 3
INPUT_FILE = 'data/train_example.bson'
DEST = 'data_example'
# INPUT_FILE = 'data/train.bson'
# DEST = 'data'

data_generator = bson.decode_file_iter(open(INPUT_FILE, 'rb'))

start = time.time()

obs_count = 0
while True:
    try:
        data_dict = data_generator.next()
    except StopIteration:
        break

    product_id = data_dict['_id']
    category_id = data_dict.get('category_id', np.nan)

    if not os.path.exists(DEST + '/' + str(category_id)):
        os.makedirs(DEST + '/' + str(category_id))

    sku_pic_index = 0
    for e, pic in enumerate(data_dict['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        toimage(picture).save(DEST + '/' + str(category_id) + '/' + str(product_id) + '_' + str(sku_pic_index) + '.jpg')
        sku_pic_index += 1

    obs_count += 1

print 'Images read/written in {} seconds'.format(time.time() - start)


dirs = os.listdir(DEST + '/validate')
for dir in dirs:
    files = os.listdir(DEST + '/validate/' + dir)
    for file in files:
        os.rename(DEST + '/validate/' + dir + '/' + file, DEST + '/train/' + dir + '/' + file)