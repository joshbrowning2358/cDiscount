import os
import shutil

bottleneck_dir = '/Users/joshuabrowning/Personal/Kaggle/cDiscount/tf_files/bottlenecks/'
input_dir = '/Users/joshuabrowning/Personal/Kaggle/cDiscount/data/train/'

input_categories = os.listdir(input_dir)
tried_cnt = 0
removed_cnt = 0
diffs = []
missing_images = 0
for category in input_categories:
    if tried_cnt % 25 == 1:
        print('Removed {} directories ({}% of total)'.format(removed_cnt, round(100 * removed_cnt / tried_cnt, 2)))
    try:
        bottleneck_images = os.listdir(bottleneck_dir + category)
        input_images = os.listdir(input_dir + category)
        if len(bottleneck_images) == len(input_images):
            shutil.rmtree(input_dir + category)
            removed_cnt += 1
        else:
            diffs += [(category, len(input_images), len(bottleneck_images))]
        tried_cnt += 1
    except OSError:
        missing_images += len(os.listdir(input_dir + category))
        tried_cnt += 1
        continue
