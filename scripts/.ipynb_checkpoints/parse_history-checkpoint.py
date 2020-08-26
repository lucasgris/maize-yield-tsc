import random
import glob
import sys

import pandas as pd

from os import listdir
from os.path import basename, dirname, splitext, join

data_dir = sys.argv[1]
key = sys.argv[2]
output = sys.argv[3]
print(data_dir, key, output)

data = dict()

for hist_file in listdir(data_dir):
    if splitext(hist_file)[1] == '.csv':
        hist_pd = pd.read_csv(join(data_dir, hist_file))
        data[hist_file.split('.csv')[0]] = hist_pd[key]

data = pd.DataFrame.from_dict(data)
data.to_csv(f'{output}_{key}.csv')