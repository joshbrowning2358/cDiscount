import os
import re
import subprocess
import shutil
import fileinput

import numpy as np

n_files = 10
chunk_location = '/Users/joshuabrowning/Personal/Kaggle/cDiscount/tf_files/bottlenecks_small_example'

current_dirs = os.listdir(chunk_location)
current_dirs = [x for x in current_dirs if not re.match('.*txt', x)]

# Convert directories to single text files
for dir in current_dirs:
    base = '{}/{}'.format(chunk_location, dir)
    filenames = [base + '/' + x for x in os.listdir(base)]
    with open(base + '.txt', 'w') as f_out:
        f_in = fileinput.input(filenames)
        for line in f_in:
            f_out.write(line + '\n')
        f_in.close()
    shutil.rmtree('{}/{}'.format(chunk_location, dir))

current_files = os.listdir(chunk_location)
current_files = [x for x in current_files if re.match('^1.*txt', x)]

for category_file in current_files:
    try:
        data = np.genfromtxt(chunk_location + '/' + category_file, delimiter=',')
    except ValueError:
        print('Failed for {}'.format(category_file))
        continue
    split_indices = np.round(np.arange(0, len(data), float(len(data))/10))
    split_indices = np.append(split_indices, len(data))
    for i in range(n_files):
        subset = data[int(split_indices[i]):int(split_indices[i+1])]
        target = int(re.sub('\.txt', '', category_file))
        labels = np.reshape(np.array([target]*subset.shape[0]), (subset.shape[0], 1))
        with open('{}/chunked_file_{}.txt'.format(chunk_location, i), 'ab') as f:
            np.savetxt(f, subset, fmt='%5.5f')
        with open('{}/chunked_labels_{}.txt'.format(chunk_location, i), 'ab') as f:
            np.savetxt(f, labels)
    os.remove('{}/{}'.format(chunk_location, category_file))
    print('Converted file {}'.format(category_file))

