import random
import glob
import sys

import pandas as pd

from os import listdir
from os.path import basename, dirname, splitext, join

data_dir = sys.argv[1]


def get_instances(data_dir):
    return set([splitext(basename(p))[0].split('_B')[0]
                for p in listdir(data_dir)])


out_name = basename(dirname(data_dir))
thermal_data_csv = open(out_name + "_thermal_data.csv", 'w')
multiespectral_data_csv = open(out_name + "_multiespectral_data.csv", 'w')

thermal_data_csv.write('Instance,Crop,B1File,Name,REP,BLK,PLOT,ENTRY,Trial,'
                       'Eval,Yield\n')
multiespectral_data_csv.write('Instance,Crop,B1File,B2File,B3File,B4File,Name,'
                              'REP,BLK,PLOT,ENTRY,Trial,Eval,Yield\n')

for instance in get_instances(data_dir):
    with open(join(data_dir, instance + ".csv")) as csv:
        data = csv.readlines()[1].rstrip().split(',')
        inst = {
            "Name"  : data[0].strip('"'),
            "Crop"  : basename(dirname(data_dir)),
            "REP"   : int(data[1]),
            "BLK"   : int(data[2]),
            "PLOT"  : int(data[3]),
            "ENTRY" : int(data[4]),
            "trial" : data[5].strip('"'),
            "Eval"  : float(data[6]),
            "Yield" : float(data[7]),
        }
        if instance[0] == 'a':
            thermal_data_csv.write(
                f"{instance},"
                f"{inst['Crop']},"
                f"{instance}_B1.tif,"
                f"{inst['Name']},"
                f"{inst['REP']},"
                f"{inst['BLK']},"
                f"{inst['PLOT']},"
                f"{inst['ENTRY']},"
                f"{inst['trial']},"
                f"{inst['Eval']},"
                f"{inst['Yield']}\n"
            )
        else:
            multiespectral_data_csv.write(
                f"{instance},"
                f"{inst['Crop']},"
                f"{instance}_B1.tif,"
                f"{instance}_B2.tif,"
                f"{instance}_B3.tif,"
                f"{instance}_B4.tif,"
                f"{inst['Name']},"
                f"{inst['REP']},"
                f"{inst['BLK']},"
                f"{inst['PLOT']},"
                f"{inst['ENTRY']},"
                f"{inst['trial']},"
                f"{inst['Eval']},"
                f"{inst['Yield']}\n"
            )
