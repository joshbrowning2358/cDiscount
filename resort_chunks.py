import os
import re
import subprocess
import shutil

import numpy as np

n_files = 10
chunk_location = '/Users/joshuabrowning/Desktop/test_tf_files'

current_dirs = os.listdir(chunk_location)
current_dirs = [x for x in current_dirs if not re.match('.*txt', x)]

# Convert directories to single text files
for dir in current_dirs:
    # subprocess.check_output(['awk', 'FNR==0{print ""}1', '{}/{}/*.txt'.format(chunk_location, dir), '>', '{}.txt'.format(dir)],
    #                         stderr=subprocess.STDOUT)
    success = True
    for i in range(10):
        result = os.system("""awk 'FNR==0{print ""}1'""" +
                           ' {}/{}/*{}\_?.jpg*.txt > {}/{}_{}.txt'.format(chunk_location, dir, i, chunk_location, i, dir))
        if result != 0:
            success = False
    if success:
        os.system('cat {}/*_{}.txt > {}/{}.txt'.format(chunk_location, dir, chunk_location, dir))
        shutil.rmtree('{}/{}'.format(chunk_location, dir))
        print('Converted directory {}/ into file {}.csv'.format(dir, dir))
    else:
        print('Conversion failed for directory {}, not removed!'.format(dir))
    os.system('rm {}/[0-9]_{}.txt'.format(chunk_location, dir))

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
        with open('{}/chunked_file_{}.txt'.format(chunk_location, i), 'ab') as f:
            np.savetxt(f, data[int(split_indices[i]):int(split_indices[i+1])], fmt='%5.5f')
    os.remove('{}/{}'.format(chunk_location, category_file))