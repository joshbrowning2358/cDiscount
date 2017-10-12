import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import pandas as pd
import numpy as np
from PIL import Image


# Simple data processing

data = bson.decode_file_iter(open('data/train_example.bson', 'rb'))

prod_to_category = dict()
pictures = []

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    prod_to_category[product_id] = category_id
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        pictures += [picture]
        # do something with the picture, etc

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

all_pics = np.zeros((1800, 1800, 3))
for i in range(100):
    row = i // 10
    col = i % 10
    all_pics[(180*row):(180*row + 180), (180*col):(180*col + 180), :] = pictures[i]
plt.imshow(all_pics, interpolation='nearest')
plt.show()
