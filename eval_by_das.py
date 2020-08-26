from models import *
import util
from livelossplot import PlotLossesKeras
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import r2_score
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
import pandas as pd
import numpy as np
import glob
from os.path import isfile, join
from os import listdir
from datetime import datetime
import gc
import random
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from trainutils import *

COLORS = cm.winter(np.linspace(0, 1, 8))
MARKERS = ['o', '*', 'x', '+', 'H', 'd', '^', 'p']

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices(
            'GPU')
        print(len(gpus), "Physical GPUs,", len(
            logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def R_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(
        tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def search_best_checkpoint(model_name, pick=-1):
    models_list = os.listdir('models')
    search = [candidate for candidate in models_list if model_name in candidate]
    print("Found", len(search), "checkpoint(s) with this name:", model_name)
    if len(search) < 1: 
        return None
    return search[pick]


def load_data(test_metadata, **dataset_kw):
    dataset = DASDataset(test_metadata, **dataset_kw)
    dataset.load_train_val_test_data()
    test_features, test_das, test_y = dataset.test_data
    return test_features, test_das, test_y


def evaluate_model_by_das(model, metadata, das, flatten=False, **dataset_kw):
    # print(das)
    test_metadata = metadata[metadata['DAS'] == das]
    features, das, y = load_data(test_metadata, **dataset_kw)
    if flatten:
        features = np.reshape(np.asarray(
            features), (len(features), features.shape[1]*features.shape[2]*features.shape[3]))
    loss, mse, r2, R2, p = model.evaluate([features, das], y, verbose=0)
    predictions = model.predict([features, das])
    return loss, mse, r2, R2, p, predictions, y


def evaluate_model(model, metadata, flatten=False, **dataset_kw):
    features, das, y = load_data(metadata, **dataset_kw)
    if flatten:
        features = np.reshape(np.asarray(
            features), (len(features), features.shape[1]*features.shape[2]*features.shape[3]))
    loss, mse, _, R2, p = model.evaluate([features, das], y, verbose=0)
    predictions = model.predict([features, das])
    r2 = r2_score(y, predictions)
    return loss, mse, r2, R2, p, predictions, y


def evaluate_best_checkpoint(model_name, pick, trial, metadata, das_list, original_das_list,**dataset_kw):
    print(dataset_kw)
    best_checkpoint = search_best_checkpoint(model_name, pick)
    checkpoint_filepath = os.path.join("models", best_checkpoint)
    model = load_model(checkpoint_filepath,
                    custom_objects={
                        "r2_keras": r2_keras,
                        "R_squared": R_squared,
                        "pearson_r": pearson_r
                    })
    
    os.makedirs(f'eval_by_das_preds/{model_name}', exist_ok=True)
    
    with open(f"eval_by_das_{trial}.csv", "a") as report:
        report.write(f'{model_name},{trial}')
        if best_checkpoint is None:
            report.write('Não foi possível encontrar modelo\n')
            return

        loss, mse, r2, R2, p, predictions, y_true = evaluate_model(model, metadata, flatten=True if 'MLP' in model_name else False, **dataset_kw)
        report.write(f',{loss},{mse},{r2},{R2},{p}')

        lines = []
        plt.figure(figsize=(10,10))
        for das, odas, color, marker in zip(das_list, original_das_list, COLORS, MARKERS):
            loss, mse, r2, R2, p, predictions, y_true = evaluate_model_by_das(model, metadata, das, flatten=True if 'MLP' in model_name else False, **dataset_kw)

            report.write(f',{loss},{mse},{r2},{R2},{p}')

            pred_csv_file = f'eval_by_das_preds/{model_name}/preds_of_das_{odas}.csv'
            with open(pred_csv_file, mode='w') as f:
                f.write("TRUE,PREDICTED\n")
                for y, y_pred in zip(y_true, predictions):
                    f.write(f"{y},{y_pred[0]}\n")

            lines.append(Line2D([0], [0], color=color, marker=marker, lw=1) )
            plt.scatter(y_true, predictions, marker=marker, s=75, color=color)
        
        report.write('\n')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.legend(lines, [str(i) for i in original_das_list])
        _ = plt.plot([-100, 100], [-100, 100])
        
        # plt.savefig(f'disp.png')
        
        plt.savefig(f'eval_by_das_preds/{model_name}/disp.png')
        plt.close()


if __name__ == "__main__":

    model_name = sys.argv[1]
    
    # pick = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    pick = -1

    if 'simple' in model_name or 'default' in model_name:
        DSHAPE = (53, 53)
    elif 'advanced' in model_name:
        DSHAPE = (80, 80)
    else:
        DSHAPE = (43, 53)

    IM_SHAPE = (DSHAPE[0], DSHAPE[1], 4)
    FLATTEN_SHAPE = IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2]


    loader_kws = {
        'im_shape': IM_SHAPE,
        'augment_type': None,
        'verbose_level': 0
    }

    dfs = dict()
    for f in os.listdir("data"):
        if os.path.splitext(f)[-1] == '.csv':
            dfs[f] = pd.read_csv(os.path.join("data", f))

    frames = []
    for k in dfs:
        if 'b2s' in k and 'multiespectral' in k:
            frames.append(dfs[k])
    metadata = pd.concat(frames)

    def extract_date(token):
        date = ''
        for s in token:
            if s.isdigit():
                date += s
        return datetime.strptime(date, '%y%m%d')

    metadata['Date'] = metadata.apply(
        lambda x: extract_date(x['Instance'].split('_')[0]), axis=1)
    metadata['Year'] = metadata.apply(lambda x: x.Date.year, axis=1)

    metadata = metadata.sort_values(['Name', 'Crop', 'Date'])
    metadata['Name'] = metadata.apply(
        lambda x: f'{(x.Name.split("_")[0])}_{int(x.Name.split("_")[1]):03}', axis=1)
    metadata['Instance'] = metadata.apply(
        lambda x: f'{x.Name}_{x.Crop}_{x.Date.year}{x.Date.month}{x.Date.day}', axis=1)

    for key in ['B1File', 'B2File', 'B3File', 'B4File']:
        metadata[key] = metadata.apply(lambda x: os.path.join(
            'data', 'RAW', x['Crop'], x[key]), axis=1)

    sowing_2016 = datetime.strptime('2016-01-19', '%Y-%m-%d')
    sowing_2017 = datetime.strptime('2017-01-24', '%Y-%m-%d')

    def get_das(date):
        if date.year == 2016:
            return (date - sowing_2016).days
        elif date.year == 2017:
            return (date - sowing_2017).days

    metadata['DAS'] = metadata.apply(lambda x: get_das(x.Date), axis=1)
    metadata['oDAS'] = metadata['DAS']
    metadata['DAS'] = (metadata['DAS']-metadata['DAS'].min()) / \
        (metadata['DAS'].max()-metadata['DAS'].min())
    metadata['DAS']

    trial = ''
    if 'TRIAL_1' in model_name:
        metadata = metadata[metadata.Trial != 'Fungicide']
        test_metadata = metadata[metadata.Year == 2017]
        trial += 'TRIAL_1'
    elif 'TRIAL_2' in model_name:
        metadata = metadata[metadata.Trial != 'Fungicide']
        test_metadata = metadata[metadata.REP == 3]
        trial += 'TRIAL_2'
    elif 'TRIAL_3' in model_name:
        test_metadata = metadata[metadata.Trial == 'Fungicide']
        trial += 'TRIAL_3'
    if '--all' in sys.argv:
        test_metadata = metadata

    train_metadata = val_metadata = test_metadata
    das_list = np.unique(test_metadata['DAS'])
    original_das_list = np.unique(test_metadata['oDAS']) 

    if 'Date' in model_name:
        NORMALIZE_BY = 'Date'
    elif 'REP' in model_name:
        NORMALIZE_BY = 'REP'
    else: 
        NORMALIZE_BY = None
    
    if 'std' in model_name:
        NORMALIZATION_TYPE = 'std'
    else: 
        NORMALIZATION_TYPE = None

    if "Yield" in model_name:
        TARGET = 'Yield'
    else:
        TARGET = 'Eval' 

    dataset_kw = {
        # "train_metadata": train_metadata,
        # "val_metadata": val_metadata,
        # "test_metadata": test_metadata,
        "loader_fn": util.process,
        "normalize_by": NORMALIZE_BY,
        "normalization_type": NORMALIZATION_TYPE,
        "loader_kw": loader_kws,
        "targets": TARGET
    }

    if not os.path.isfile(f"eval_by_das_{trial}.csv"):
        with open(f"eval_by_das_{trial}.csv", "w") as report:
            report.write('model_name,trial,loss,mse,R2_skl,R2,p')
            for das in original_das_list:
                report.write(
                    f',{das}_loss,{das}_mse,{das}_R2_skl,{das}_R2,{das}_p')
            report.write('\n')
    evaluate_best_checkpoint(model_name, pick, trial, test_metadata, das_list, original_das_list, **dataset_kw)
