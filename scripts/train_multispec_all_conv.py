import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from models import *
import util
from livelossplot import PlotLossesKeras
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, BatchNormalization,
                                     MaxPooling2D, GlobalAveragePooling2D, Input,
                                     Activation, Flatten)
from tensorflow.keras.models import Sequential, load_model, Model
import pandas as pd
import numpy as np
import datetime
import random
import glob
import os
from os import listdir
from datetime import datetime
DATASET = 'b2s'
TYPE = "multiespectral"
SHOW_INFO = False

SHAPE = (53, 53)

BALANCE_DATA = True


class Generator(Sequence):
    def __init__(self, metadata, loader_fn, loader_kw, target, new_shape=None,
                 batch_size: int = 32, augmentation=1, shuffle: bool = True,
                 temp_dir=None, shape=None, use_cache=False):
        self.metadata = metadata
        self._batch_size = batch_size
        self._loader = loader_fn
        self._loader_kw = loader_kw
        self._new_shape = new_shape
        self._augmentation = augmentation
        if shape is not None:
            self._shape = shape
        else:
            self._set_shape()
        self._target = target
        self._temp_dir = temp_dir
        self._use_cache = use_cache
        self._cache = {}
        if temp_dir is not None:
            os.makedirs(temp_dir, exist_ok=True)
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))

        if shuffle:
            self.metadata = self.metadata.sample(frac=1)
    
    def _set_shape(self):
        sample = self.metadata.iloc[0, ]
        _, data = self._loader(sample, **self._loader_kw)
        if len(data) > 1 and self._augmentation == 1:
            print(
                f"Warning: augmented data detected. Setting augmentation to {len(data)}")
            self._augmentation = len(data)
        # data must be in shape (augmented, data shape ...)
        # where augmented must be at least equal to 1, when no augmentation
        # is performed
        self._shape = (np.asarray(data).shape)
    

    def _is_saved(self, item):
        saved = f'{item.Instance}_{self._shape}_{self._augmentation}.npy'
        if not os.path.isfile(os.path.join(self._temp_dir, saved)):
            return False
        return True
    

    def _load_from_disk(self, item):
        saved = f'{item.Instance}_{self._shape}_{self._augmentation}.npy'
        return np.load(os.path.join(self._temp_dir, saved))
    

    def _save_to_disk(self, item, data):
        saved = f'{item.Instance}_{self._shape}_{self._augmentation}.npy'
        np.save(os.path.join(self._temp_dir, saved), arr=data)


    def __getitem__(self, index):
        items = self.metadata.iloc[index*(self._batch_size // self._augmentation):
                                   (index+1)*(self._batch_size // self._augmentation), ]
        # Fill batches
        x = []
        y = []
        for i, item in items.iterrows():
            data = None
            if self._use_cache and item.Instance in self._cache:
                data = self._cache[item.Instance]

            if data is None and self._temp_dir is not None and self._is_saved(item) :
                data = self._load_from_disk(item)
            
            if data is None:
                _, data = self._loader(item, **self._loader_kw)
                if self._temp_dir is not None:
                    self._save_to_disk(item, data)
                if self._use_cache:
                    self._cache[item.Instance] = data
            for d in data:
                if self._new_shape is not None:
                    x.append(d.reshape(self._new_shape))
                else:
                    x.append(d)
                y.append(item[self._target])
        return np.asarray(x), np.asarray(y)

    def __len__(self):
        return int((np.floor(self.metadata.shape[0])*self._augmentation) / self._batch_size)


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


def run(augment, epoch, lr, lossfunction, topology):
    TARGET = 'Eval'
    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
    AUGMENT = augment

    AUG_INC = 1
    if AUGMENT is None:
        AUG_INC = 1
    elif AUGMENT == 'simple':
        AUG_INC = 2
    elif AUGMENT == 'default':
        AUG_INC = 4
    elif AUGMENT == 'default' and SHAPE[0] == SHAPE[1]:
        AUG_INC = 6
    elif AUGMENT == 'advanced':
        AUG_INC = 16
    elif AUGMENT == 'advanced' and SHAPE[0] == SHAPE[1]:
        AUG_INC = 18

    EPOCHS = epoch
    LOSSF = lossfunction
    LR = lr
    BATCH_SIZE = 36
    IM_SHAPE = (SHAPE[0], SHAPE[1], 4)
    FLATTEN_SHAPE = IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2]
    TOPOLOGY = topology
    MODEL_NAME = f'{TOPOLOGY}_{DATASET}_{TYPE}_{TARGET}'

    dfs = dict()
    for f in os.listdir("data"):
        if os.path.splitext(f)[-1] == '.csv':
            dfs[f] = pd.read_csv(os.path.join("data", f))

    frames = []
    for k in dfs:
        if DATASET in k and TYPE in k:
            frames.append(dfs[k])
    metadata = pd.concat(frames)
    metadata.head(10)

    for key in ['B1File', 'B2File', 'B3File', 'B4File']:
        metadata[key] = metadata.apply(lambda x: os.path.join(
            'data', 'RAW', x['Crop'], x[key]), axis=1)
    metadata.head(10)

    train_gen = Generator(metadata=metadata, batch_size=BATCH_SIZE,
                          loader_fn=util.process,
                          augmentation=AUG_INC,
                          target=TARGET,
                          loader_kw={"im_shape": IM_SHAPE,
                                     "augment_type": AUGMENT,
                                     "verbose_level": 0})

    if SHOW_INFO:
        print("Tamanho do dataset:", len(metadata))

    if SHOW_INFO:
        temp = metadata['Trial'].value_counts()
        labels = temp.index
        sizes = (temp / temp.sum())*100
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')

    if SHOW_INFO:
        plt.figure(figsize=(15, 8))
        evals = metadata['Eval'].value_counts()
        sns.barplot(evals.index, evals.values)
        plt.xticks(rotation='vertical')
        plt.xlabel('Pontuação')
        plt.ylabel('Frequência')
        plt.title("Evaluações no dataset")

    metadata = metadata.sample(frac=1, random_state=10)
    metadata["Split"] = np.asarray(["Train" if i <= len(metadata)*.8 else
                                    "Test" if i < (len(metadata)*.8 + len(metadata)*.1) else
                                    "Validation" for i in range(len(metadata))])

    metadata.head()

    train_metadata = metadata.loc[metadata['Split'] == "Train"]
    val_metadata = metadata.loc[metadata['Split'] == "Validation"]
    test_metadata = metadata.loc[metadata['Split'] == "Test"]
    print("Quantidade de instâncias (sem aumento de dados) totais:",
          len(metadata))
    print("Quantidade de instâncias (sem aumento de dados) para treinamento:",
          len(train_metadata))
    print("Quantidade de instâncias (sem aumento de dados) para validação:",
          len(val_metadata))
    print("Quantidade de instâncias (sem aumento de dados) para teste:",
          len(test_metadata))

    train_set = set(train_metadata.Instance)
    val_set = set(val_metadata.Instance)
    test_set = set(test_metadata.Instance)
    for i in val_set:
        assert not (i in train_set or i in test_set)
    for i in test_set:
        assert not (i in train_set or i in val_set)

    if BALANCE_DATA:
        g = train_metadata.groupby('Eval', as_index=False)
        metadata_bal = pd.DataFrame(g.apply(lambda x: x.sample(g.size().max(),
                                                               replace=True).reset_index(drop=True)))
        metadata_bal.reset_index(drop=True, inplace=True)
        train_metadata = metadata_bal

    train_features = []
    train_y = []
    print("Loading trainning data")
    for index, train_data in train_metadata.iterrows():
        index, data = util.process(train_data,
                                   im_shape=IM_SHAPE,
                                   augment_type=None,
                                   verbose_level=0)

        train_features.append(data[0])
        train_y.append(train_data[TARGET])

    val_features = []
    val_y = []
    print("Loading validation data")
    for index, val_data in val_metadata.iterrows():
        index, data = util.process(val_data,
                                   im_shape=IM_SHAPE,
                                   augment_type=None,
                                   verbose_level=0)

        val_features.append(data[0])
        val_y.append(val_data[TARGET])

    print("Loading test data")
    test_features = []
    test_y = []
    for index, test_data in test_metadata.iterrows():
        index, data = util.process(test_data,
                                   im_shape=IM_SHAPE,
                                   augment_type=None,
                                   verbose_level=0)
        test_features.append(data[0])
        test_y.append(test_data[TARGET])

    train_features = np.asarray(train_features)
    train_y = np.asarray(train_y)
    test_features = np.asarray(test_features)
    test_y = np.asarray(test_y)
    val_features = np.asarray(val_features)
    val_y = np.asarray(val_y)

    model = globals()[TOPOLOGY](IM_SHAPE)

    optimizer = tf.keras.optimizers.Adam(LR)

    model.compile(loss=LOSSF,
                  optimizer=optimizer,
                  metrics=['mae', 'mse', r2_keras, pearson_r])

    os.makedirs('topologias', exist_ok=True)
    plot_model(model, to_file=f'topologias{os.sep}{MODEL_NAME}.png',
               show_shapes=True, show_layer_names=False)

    model.summary()

    print(test_features.shape, test_y.shape)

    loss, mae, mse, r2, p = model.evaluate(test_features, test_y, verbose=1)
    print(f"Loss={loss}\nMAE={mae}\nMSE={mse}\nR2={r2}\nPearson={p}")

    checkpoint_filepath = f'models/{MODEL_NAME}_{NOW}.h5'

    os.makedirs('models', exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss',
            verbose=0)
    ]
    start = datetime.now()

    train_gen = Generator(metadata=train_metadata,
                          batch_size=BATCH_SIZE,
                          loader_fn=util.process,
                          augmentation=AUG_INC,
                          target=TARGET,
                          shape=IM_SHAPE,
                          use_cache=True,
                        #   temp_dir='__temp__',
                          loader_kw={"im_shape": IM_SHAPE,
                                     "augment_type": AUGMENT,
                                     "verbose_level": 0})

    history = model.fit(x=train_gen,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(val_features, val_y),
                        max_queue_size=10,
                        workers=7,
                        use_multiprocessing=False)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    print("*"*10, "TEST (TRAIN SET)", "*"*10)
    train_loss, train_mae, train_mse, train_r2, train_p = model.evaluate(
        train_features, train_y, verbose=1)
    print(
        f"Loss={train_loss}\nMAE={train_mae}\nMSE={train_mse}\nR2={train_r2}\nPearson={train_p}")

    print("*"*10, "TEST (VAL SET)", "*"*10)
    val_loss, val_mae, val_mse, val_r2, val_p = model.evaluate(
        val_features, val_y, verbose=1)
    print(
        f"Loss={val_loss}\nMAE={val_mae}\nMSE={val_mse}\nR2={val_r2}\nPearson={val_p}")

    print("*"*10, "TEST (TEST SET)", "*"*10)
    test_loss, test_mae, test_mse, test_r2, test_p = model.evaluate(
        test_features, test_y, verbose=1)
    print(
        f"Loss={test_loss}\nMAE={test_mae}\nMSE={test_mse}\nR2={test_r2}\nPearson={test_p}")

    model = load_model(checkpoint_filepath, custom_objects={
                       "r2_keras": r2_keras, "pearson_r": pearson_r})

    print("*"*10, "TEST (TRAIN SET)", "*"*10)
    train_loss, train_mae, train_mse, train_r2, train_p = model.evaluate(
        train_features, train_y, verbose=1)
    print(
        f"Loss={train_loss}\nMAE={train_mae}\nMSE={train_mse}\nR2={train_r2}\nPearson={train_p}")

    print("*"*10, "TEST (VAL SET)", "*"*10)
    val_loss, val_mae, val_mse, val_r2, val_p = model.evaluate(
        val_features, val_y, verbose=1)
    print(
        f"Loss={val_loss}\nMAE={val_mae}\nMSE={val_mse}\nR2={val_r2}\nPearson={val_p}")

    print("*"*10, "TEST (TEST SET)", "*"*10)
    test_loss, test_mae, test_mse, test_r2, test_p = model.evaluate(
        test_features, test_y, verbose=1)
    print(
        f"Loss={test_loss}\nMAE={test_mae}\nMSE={test_mse}\nR2={test_r2}\nPearson={test_p}")

    test_predictions = model.predict(test_features)

    val_predictions = model.predict(val_features)

    train_predictions = model.predict(train_features)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    os.makedirs(f'graficos{os.sep}PREDICT', exist_ok=True)
    ax1.scatter(train_y, train_predictions, marker='.', color='r')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Train Predictions')
    ax1.axis('equal')
    ax1.axis('square')
    _ = ax1.plot([-100, 100], [-100, 100])

    ax2.scatter(val_y, val_predictions, marker='.', color='g')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Val Predictions')
    ax2.axis('equal')
    ax2.axis('square')
    _ = ax2.plot([-100, 100], [-100, 100])

    ax3.scatter(test_y, test_predictions, marker='.')
    ax3.set_xlabel('True Values')
    ax3.set_ylabel('Predictions')
    ax3.axis('equal')
    ax3.axis('square')
    _ = ax3.plot([-100, 100], [-100, 100])
    plt.savefig(f'graficos{os.sep}PREDICT{os.sep}{MODEL_NAME}_{NOW}.png')
    plt.close()

    _, ax = plt.subplots()
    ax.scatter(x=range(0, test_y.size), y=test_y,
               c='blue', label='Actual', alpha=0.3)
    ax.scatter(x=range(0, test_predictions.size), y=test_predictions,
               c='red', label='Predicted', alpha=0.3)

    plt.title('Actual and predicted values')
    plt.xlabel('Observations')
    plt.ylabel('mpg')
    plt.legend()
    plt.close()

    diff = test_y.reshape(test_predictions.shape) - test_predictions
    plt.hist(diff, bins=40)
    plt.title('Histogram of prediction errors')
    plt.xlabel('MPG prediction error')
    plt.ylabel('Frequency')
    plt.close()

    os.makedirs(f'graficos{os.sep}MAE', exist_ok=True)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'graficos{os.sep}MAE{os.sep}{MODEL_NAME}_{NOW}.png')
    plt.close()

    os.makedirs(f'graficos{os.sep}LOSS', exist_ok=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'graficos{os.sep}LOSS{os.sep}{MODEL_NAME}_{NOW}.png')
    plt.close()

    os.makedirs(f'graficos{os.sep}R2', exist_ok=True)
    plt.plot(history.history['r2_keras'])
    plt.plot(history.history['val_r2_keras'])
    plt.title('R2')
    plt.ylabel('R2')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'graficos{os.sep}R2{os.sep}{MODEL_NAME}_{NOW}.png')
    plt.close()

    os.makedirs(f'graficos{os.sep}Pearson', exist_ok=True)
    plt.plot(history.history['pearson_r'])
    plt.plot(history.history['val_pearson_r'])
    plt.title('Pearson')
    plt.ylabel('Pearson')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'graficos{os.sep}Pearson{os.sep}{MODEL_NAME}_{NOW}.png')
    plt.close()

    hist_df = pd.DataFrame(history.history)

    os.makedirs('history', exist_ok=True)
    hist_csv_file = f'history/{MODEL_NAME}_{NOW}.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    if not os.path.isfile('report.csv'):
        with open('report.csv', 'w') as report:
            report.write('DATA,NOME,TOPOLOGIA,DATASET,TARGET,AUGMENTATION,EPOCAS,LR,'
                         'LOSS,TEST_LOSS,TEST_MAE,TEST_MSE,TEST_PEARSON,TEST_R2\n')
    with open('report.csv', 'a') as report:
        report.write(f'{NOW},{MODEL_NAME},{TOPOLOGY},{DATASET},'
                     f'{TARGET},{AUGMENT},{EPOCHS},{LR},{LOSSF},'
                     f'{test_loss},{test_mae},{test_mse},{test_p},{test_r2}\n')


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:

            print(e)

    augments = [None, 'simple', 'default', 'advanced']
    epochs = [100]
    lrs = [0.01, 0.001, 0.0001]
    lossfunctions = ['mean_squared_logarithmic_error', 'mse']
    topologies = ['CNN_UltraMegaShallow_SmallKernel',
                  'CNN_UltraShallow_SmallKernel',
                  'CNN_BatchNorm',
                  'CNN_SmallKernel_BatchNorm',
                  'CNN',
                  'CNN_SmallKernel',
                  'CNN_drop1',
                  'CNN_drop2']

    random.shuffle(augments)
    random.shuffle(lrs)
    random.shuffle(lossfunctions)
    random.shuffle(topologies)

    for augment in augments:
        for epoch in epochs:
            for lr in lrs:
                for lossfunction in lossfunctions:
                    for topology in topologies:
                        try:
                            run(augment=augment,
                                epoch=epoch,
                                lr=lr,
                                lossfunction=lossfunction,
                                topology=topology)
                        except Exception as e:
                            with open('errors.txt', 'a') as outerr:
                                outerr.write(f'{augment}, {epoch},'
                                                f'{lr}, {lossfunction}, {topology},'
                                                f': {str(e)}\n')
                            continue
