
from trainutils import *
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
import time
from os.path import isfile, join
from os import listdir
from datetime import datetime
import gc
import random
import sys
import os

DATASET = 'b2s'  # b2s ou bfs

# SPLITS = [1, 2, 3]

TRANSFORMATIONS = {
    'NVDI': util.ndvi,
    'RDVI': util.rdvi,
    'OSAVI': util.osavi,
    'MSR': util.msr,
    'MCARI1': util.mcari1,
    'MCARI2': util.mcari2,
    'PSSRA': util.pssra,
    'G': util.g_rgb_ratio,
    "None": None,
    None: None
}

AUGMENTATIONS = [None]

TYPE = "thermal"

TOPOLOGIES = [
    'CNN_DAS_leakyrelu_heavy2_pool',
    'CNN_DAS_leakyrelu',
    'CNN_DAS_leakyrelu_heavy3_pool2',
    'CNN_DAS_leakyrelu2',
    'CNN_DAS_leakyrelu3',
    'CNN_DAS_leakyrelu4',
    'CNN_DAS_smallkernel_leakyrelu',
    'CNN_DAS_leakyrelu_pool',
    'CNN_DAS_leakyrelu_heavy',
    'MLP_DAS_leakyrelu',
    'MLP_DAS_leakyrelu2'
]


def run(**kwargs):
    print(COMMENTS)

    for k in kwargs:
        print(k, kwargs[k])

    for i in range(3):
        time.sleep(1)
    
    gc.collect(generation=2)

    trainned = False
    if os.path.isfile((f'reports{os.sep}TRIAL{TRIAL}{os.sep}report.csv')):
        with open(f'reports{os.sep}TRIAL{TRIAL}{os.sep}report.csv', 'r') as report:
            for line in report.readlines():
                tokens = line.split(',')
                if (str(TOPOLOGY) in tokens and str(AUGMENT) in tokens and str(TRANSFORMATION) in tokens and str(TYPE) in tokens):
                    print("Skipping", str(TOPOLOGY), str(AUGMENT))
                    trainned = True
    if trainned:
        return
    print("Trainning", str(TOPOLOGY), str(
        AUGMENT), str(TRANSFORMATION), str(TYPE))

    # In[2]:

    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

    # In[3]:

    # Set a optmized shape
    if AUGMENT == 'simple' or AUGMENT == 'default':
        DSHAPE = (30, 30)
    elif AUGMENT == 'advanced':
        DSHAPE = (50, 50)
    else:
        DSHAPE = (24, 24)
        
    AUG_INC = 1
    if AUGMENT is None:
        AUG_INC = 1
    elif AUGMENT == 'simple':
        AUG_INC = 2
    elif AUGMENT == 'default':
        AUG_INC = 4
    elif AUGMENT == 'default' and DSHAPE[0] == DSHAPE[1]:
        AUG_INC = 6
    elif AUGMENT == 'advanced':
        AUG_INC = 16
    elif AUGMENT == 'advanced' and DSHAPE[0] == DSHAPE[1]:
        AUG_INC = 18

    # ### Definições de treinamento

    # In[4]:

    EPOCHS = 1200
    LR = 0.01
    LOSSF = 'mse'  # mean_squared_logarithmic_error mse
    BATCH_SIZE = 45
    if TRANSFORMATION is not None or TYPE == 'thermal':
        IM_SHAPE = (DSHAPE[0], DSHAPE[1], 1)
    else:
        IM_SHAPE = (DSHAPE[0], DSHAPE[1], 4)
    FLATTEN_SHAPE = IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2]
    # TOPOLOGY = 'MLP_DAS_dropInc'
    MODEL_NAME = f'{TOPOLOGY}_{DATASET}_{TYPE}_{TRANSFORMATION}_{TARGET}_NORMALIZE_BY_{NORMALIZE_BY}_NORMALIZATION_TYPE_{NORMALIZATION_TYPE}_TRIAL_{TRIAL}_AUG_{AUGMENT}_LR{LR}_BS{BATCH_SIZE}_BAL_{BALANCE_DATA}'

    os.makedirs(f'reports{os.sep}TRIAL{TRIAL}', exist_ok=True)
    with open(f'reports/TRIAL{TRIAL}/notes_{MODEL_NAME}.md', 'w') as f:
        f.write(COMMENTS)

    # In[5]:

    if 'CNN' in TOPOLOGY:
        SHAPE = IM_SHAPE
    else:
        SHAPE = FLATTEN_SHAPE

    # Necessário na minha máquina. Estava ocorrendo um erro devido à GPU e esse código resolveu.

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

    # ## Funções para carregar os dados

    # ### Metadados

    # In[ ]:

    dfs = dict()
    for f in os.listdir("data"):
        if os.path.splitext(f)[-1] == '.csv':
            dfs[f] = pd.read_csv(os.path.join("data", f))

    frames = []
    for k in dfs:
        if DATASET in k and TYPE in k:
            frames.append(dfs[k])
    metadata = pd.concat(frames)
    metadata.head(3)

    # #### Adicionando data

    # In[ ]:

    def extract_date(token):
        date = ''
        for s in token:
            if s.isdigit():
                date += s
        return datetime.strptime(date, '%y%m%d')

    # In[ ]:

    metadata['Date'] = metadata.apply(
        lambda x: extract_date(x['Instance'].split('_')[0]), axis=1)
    metadata['Year'] = metadata.apply(lambda x: x.Date.year, axis=1)
    metadata.head(3)

    metadata['TSC'] = metadata['Eval']
    # #### Ordenando por nome,crop,data

    # In[ ]:

    metadata = metadata.sort_values(['Name', 'Crop', 'Date'])
    metadata.head(3)

    # #### Definindo a instância

    # Neste experimento vamos considerar cada plot individual como uma instância. Um plot é o conjunto de 4 diferentes bandas espectrais de uma data específica.

    # In[ ]:

    metadata['Name'] = metadata.apply(
        lambda x: f'{(x.Name.split("_")[0])}_{int(x.Name.split("_")[1]):03}', axis=1)

    # In[ ]:

    metadata['Instance'] = metadata.apply(
        lambda x: f'{x.Name}_{x.Crop}_{x.Date.year}{x.Date.month}{x.Date.day}', axis=1)

    # In[ ]:

    metadata.head()

    # In[ ]:

    for key in ['B1File']:
        metadata[key] = metadata.apply(lambda x: os.path.join(
            'data', 'RAW', x['Crop'], x[key]), axis=1)
    metadata.head(3)

    # #### Definindo *Days after sowing*

    # In[ ]:

    sowing_2016 = datetime.strptime('2016-01-19', '%Y-%m-%d')
    sowing_2017 = datetime.strptime('2017-01-24', '%Y-%m-%d')

    def get_das(date):
        if date.year == 2016:
            return (date - sowing_2016).days
        elif date.year == 2017:
            return (date - sowing_2017).days

    # In[ ]:

    metadata['DAS'] = metadata.apply(lambda x: get_das(x.Date), axis=1)
    metadata.head(3)

    # In[ ]:

    if SHOW_INFO:
        plt.figure(figsize=(8, 3))
        evals = metadata['DAS'].value_counts()
        sns.barplot(evals.index, evals.values)
        plt.xticks(rotation='vertical')
        plt.xlabel('Pontuação')
        plt.ylabel('Dias')
        plt.title("Days after sowing")
        # plt\.show\(\)

    # #### Normalizando DAS

    # In[ ]:

    metadata['DAS'] = (metadata['DAS']-metadata['DAS'].min()) / \
        (metadata['DAS'].max()-metadata['DAS'].min())
    metadata['DAS']

    # ### Split dos dados

    # Nesse experimento os dados com fungicidas serão desconsiderados.

    # In[ ]:

    # In[ ]:
    if SPLIT == 1:
        metadata = metadata[metadata.Trial != 'Fungicide']
        train_metadata = metadata[metadata.Year == 2016]
        test_metadata = metadata[metadata.Year == 2017]
        if USE_TEST_AS_VAL:
            val_metadata = test_metadata
        else:
            val_metadata = train_metadata.sample(frac=.20)
            cond = val_metadata['Instance'].isin(train_metadata['Instance'])
            train_metadata.drop(val_metadata[cond].index, inplace=True)
    elif SPLIT == 2:
        metadata = metadata[metadata.Trial != 'Fungicide']
        train_metadata = metadata[metadata.REP < 3]
        test_metadata = metadata[metadata.REP == 3]
        if USE_TEST_AS_VAL:
            val_metadata = test_metadata
        else:
            val_metadata = test_metadata.sample(frac=.50)
            cond = val_metadata['Instance'].isin(test_metadata['Instance'])
            test_metadata.drop(val_metadata[cond].index, inplace=True)
    elif SPLIT == 3:
        train_metadata = metadata[metadata.Trial != 'Fungicide']
        test_metadata = metadata[metadata.Trial == 'Fungicide']
        if USE_TEST_AS_VAL:
            val_metadata = test_metadata
        else:
            val_metadata = train_metadata.sample(frac=.20)
            cond = val_metadata['Instance'].isin(train_metadata['Instance'])
            train_metadata.drop(val_metadata[cond].index, inplace=True)
    print("Tamanho dos dados de treinamento: ", len(train_metadata))
    print("Tamanho dos dados de validação: ", len(val_metadata))
    print("Tamanho dos dados de teste: ", len(test_metadata))

    # In[ ]:

    if SHOW_INFO:
        total = len(train_metadata) + \
            len(val_metadata) + len(test_metadata)
        sizes = [len(train_metadata)/total, len(val_metadata) /
                 total, len(test_metadata)/total]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=["Train", "Validation", "Test"], autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        # plt\.show\(\)

    # In[ ]:

    print("Quantidade de instâncias (sem aumento de dados) totais:",
          len(metadata))
    print("Quantidade de instâncias (sem aumento de dados) para treinamento:",
          len(train_metadata))
    print("Quantidade de instâncias (sem aumento de dados) para validação:",
          len(val_metadata))
    print("Quantidade de instâncias (sem aumento de dados) para teste:",
          len(test_metadata))

    train_set = set(train_metadata.Instance)
    val_set = set(val_metadata.Instance) if not USE_TEST_AS_VAL else set()
    test_set = set(test_metadata.Instance)
    for i in val_set:
        assert not (i in train_set or i in test_set)
    for i in test_set:
        assert not (i in train_set or i in val_set)
        # #### Carregando alguns dados
    print(TRANSFORMATIONS)
    # In[ ]:
    loader_kws = {"im_shape": IM_SHAPE,
                  "augment_type": None,
                  "verbose_level": 0,
                  "transform_f": TRANSFORMATIONS[TRANSFORMATION]}

    # train_gen = RegressorDASGenerator(metadata=metadata,
    #                                   batch_size=BATCH_SIZE,
    #                                   metadata_flow=train_metadata,
    #                                   loader_fn=util.process_one_band,
    #                                   train_metadata=train_metadata,
    #                                   val_metadata=val_metadata,
    #                                   test_metadata=test_metadata,
    #                                   augmentation=AUG_INC,
    #                                   normalize_by=NORMALIZE_BY,
    #                                   normalization_type=NORMALIZATION_TYPE,
    #                                   targets=TARGET,
    #                                   loader_kw=loader_kws)

    # #### Balanceamento dos dados

    # In[ ]:

    if BALANCE_DATA:
        g = train_metadata.groupby(TARGET, as_index=False)
        metadata_bal = pd.DataFrame(g.apply(lambda x: x.sample(g.size().max(),
                                                               replace=True).reset_index(drop=True)))
        metadata_bal.reset_index(drop=True, inplace=True)
        train_metadata = metadata_bal

    # In[ ]:

    if SHOW_INFO:
        plt.figure(figsize=(8, 3))
        evals = train_metadata[TARGET].value_counts()
        sns.barplot(evals.index, evals.values)
        plt.xticks(rotation='vertical')
        plt.xlabel('Pontuação')
        plt.ylabel('Frequência')
        plt.title("Evaluações no dataset - Treinamento")
        # plt\.show\(\)

    dataset = DASDataset(metadata,
                         train_metadata=train_metadata,
                         val_metadata=val_metadata,
                         test_metadata=test_metadata,
                         loader_fn=util.process_one_band,
                         normalize_by=NORMALIZE_BY,
                         normalization_type=NORMALIZATION_TYPE,
                         loader_kw=loader_kws,
                         targets=TARGET)
    dataset.load_train_val_test_data()

    train_features, train_das, train_y = dataset.train_data
    val_features, val_das, val_y = dataset.val_data
    test_features, test_das, test_y = dataset.test_data

    loader_kws = {"im_shape": IM_SHAPE,
                  "augment_type": AUGMENT,
                  "verbose_level": 0,
                  "transform_f": TRANSFORMATIONS[TRANSFORMATION]}
    # In[ ]:

    if SHOW_INFO:
        plt.pcolormesh(train_features[0][:, :, 0])
        plt.axis('equal')
        plt.colorbar()
        # plt\.show\(\)
        plt.close()

    # In[ ]:

    if SHAPE == FLATTEN_SHAPE:
        val_features = np.reshape(np.asarray(
            val_features), (len(val_features), FLATTEN_SHAPE))
        train_features = np.reshape(np.asarray(
            train_features), (len(train_features), FLATTEN_SHAPE))
        test_features = np.reshape(np.asarray(
            test_features), (len(test_features), FLATTEN_SHAPE))

    # ## Métricas

    # In[ ]:
    def R_squared(y_true, y_pred):
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(
            tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        r2 = tf.subtract(1.0, tf.divide(residual, total))
        return r2

    def r2_keras(y_true, y_pred):
        #     return tfa.metrics.RSquare(y_true, y_pred)

        #     RSS =  K.sum(K.square( y_true- y_pred ))
        #     TSS = K.sum(K.square( y_true - K.mean(y_true) ) )
        #     return ( 1. - RSS/(TSS) )

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

    # ## Modelos

    # **Veja models.py**

    # In[ ]:

    model = globals()[TOPOLOGY](SHAPE)

    # ### Otimizador

    # In[ ]:
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                   patience=100,
                                                                   verbose=0,
                                                                   factor=0.8,
                                                                   min_lr=0.0001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    # decay_rate = (LR / EPOCHS)
    # optimizer = tf.keras.optimizers.Adam(lr=LR, decay=decay_rate)
    # ### Compile

    # In[ ]:

    model.compile(loss=LOSSF,
                  optimizer=optimizer,
                  metrics=['mse', r2_keras, R_squared, pearson_r])

    # In[ ]:

    os.makedirs('topologias', exist_ok=True)
    plot_model(model, to_file=f'topologias{os.sep}{TOPOLOGY}.png',
               show_shapes=True, show_layer_names=False)

    # In[ ]:

    model.summary()

    # In[ ]:

    print(test_features.shape, test_das.shape, test_y.shape)

    # In[ ]:

    loss, mse, r2, R2, p = model.evaluate(
        [test_features, test_das], test_y, verbose=1)
    print(f"Loss={loss}\nMSE={mse}\nR2={r2}\nR2={R2}\nPearson={p}")

    # In[ ]:
    interv = EPOCHS//50
    plot_metrics = PlotMetricsRegressor([val_features, val_das], val_y, interv,
                                        savefig=f'{MODEL_NAME}_{NOW}', savefig_epoch=EPOCHS-interv)

    # ## Treinamento

    # In[ ]:

    checkpoint_filepath = f'models/{MODEL_NAME}_{NOW}.h5'

    os.makedirs('models', exist_ok=True)
    callbacks = [
        learning_rate_reduction,
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            # mode='max',
            monitor='val_loss',
            # monitor='val_r2_keras',
            verbose=0),
        ShowLR()
    ]
    if SHOW_INFO:
        callbacks.append(plot_metrics)
    start = datetime.now()
    with np.errstate(divide='ignore'):
        pass
    if AUGMENT is not None:
        print("Using generator")

        loader_kws = {"im_shape": IM_SHAPE,
                      "augment_type": AUGMENT,
                      "verbose_level": 0,
                      "transform_f": TRANSFORMATIONS[TRANSFORMATION]}

        train_gen = RegressorDASGenerator(metadata_flow=train_metadata,
                                          metadata=metadata,
                                          train_metadata=train_metadata,
                                          val_metadata=val_metadata,
                                          test_metadata=test_metadata,
                                          batch_size=BATCH_SIZE,
                                          loader_fn=util.process_one_band,
                                          augmentation=AUG_INC,
                                          # use_cache=True,  # BUG
                                          normalize_by=NORMALIZE_BY,
                                          normalization_type=NORMALIZATION_TYPE,
                                          targets=TARGET,
                                          new_shape=SHAPE,
                                          loader_kw=loader_kws)

        for i in train_gen:
            print(AUG_INC, AUGMENT,  loader_kws, len(i[1]))
            break

        history = model.fit(x=train_gen,
                            epochs=EPOCHS,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(
                                [val_features, val_das], val_y),
                            max_queue_size=10,
                            workers=3,
                            use_multiprocessing=True)
    else:
        history = model.fit(x=[train_features, train_das],
                            y=train_y,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=([val_features, val_das], val_y))

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # ### Teste

    # In[ ]:

    print("*"*10, "TEST (TRAIN SET)", "*"*10)
    train_loss, train_mse, train_r2, train_R2, train_p = model.evaluate(
        [train_features, train_das], train_y, verbose=1)
    print(
        f"Loss={train_loss}\nMSE={train_mse}\nr2={train_r2}\nR2={train_r2}\nPearson={train_p}")

    print("*"*10, "TEST (VAL SET)", "*"*10)
    val_loss, val_mse, val_r2, val_R2, val_p = model.evaluate(
        [val_features, val_das], val_y, verbose=1)
    print(
        f"Loss={val_loss}\nMSE={val_mse}\nr2={val_r2}\nR2={val_r2}\nPearson={val_p}")

    print("*"*10, "TEST (TEST SET)", "*"*10)
    test_loss, test_mse, test_r2, test_R2, test_p = model.evaluate(
        [test_features, test_das], test_y, verbose=1)
    print(
        f"Loss={test_loss}\nMSE={test_mse}\rR2={test_r2}\nR2={test_r2}\nPearson={test_p}")

    # ## Teste do melhor modelo

    # In[ ]:

    model = load_model(checkpoint_filepath, custom_objects={
        "r2_keras": r2_keras, "R_squared": R_squared, "pearson_r": pearson_r})

    # ### Teste

    # In[ ]:

    print("*"*10, "TEST (TRAIN SET)", "*"*10)
    train_loss, train_mse, train_r2, train_R2, train_p = model.evaluate(
        [train_features, train_das], train_y, verbose=1)
    print(
        f"Loss={train_loss}\nMSE={train_mse}\nr2={train_r2}\nR2={train_r2}\nPearson={train_p}")

    print("*"*10, "TEST (VAL SET)", "*"*10)
    val_loss, val_mse, val_r2, val_R2, val_p = model.evaluate(
        [val_features, val_das], val_y, verbose=1)
    print(
        f"Loss={val_loss}\nMSE={val_mse}\nr2={val_r2}\nR2={val_r2}\nPearson={val_p}")

    print("*"*10, "TEST (TEST SET)", "*"*10)
    test_loss, test_mse, test_r2, test_R2, test_p = model.evaluate(
        [test_features, test_das], test_y, verbose=1)
    print(
        f"Loss={test_loss}\nMSE={test_mse}\rR2={test_r2}\nR2={test_r2}\nPearson={test_p}")

    # ### Predições

    # In[ ]:

    test_predictions = model.predict([test_features, test_das])

    # In[ ]:

    val_predictions = model.predict([val_features, val_das])

    # In[ ]:

    train_predictions = model.predict([train_features, train_das])

    # In[ ]:

    train_r2 = r2_score(train_y, train_predictions)
    val_r2 = r2_score(val_y, val_predictions)
    test_r2 = r2_score(test_y, test_predictions)

    # In[ ]:

    print('R2 Train set:', r2_score(train_y, train_predictions))
    print('R2 Validation set:', r2_score(val_y, val_predictions))
    print('R2 Test set:', r2_score(test_y, test_predictions))

    # In[ ]:

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    os.makedirs(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}PREDICT', exist_ok=True)

    ax1.scatter(train_y, train_predictions, marker='.', color='r')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Train Predictions')
    ax1.axis('equal')
    ax1.axis('square')
    _ = ax1.plot([-100, 100], [-100, 100])

    ax2.scatter(val_y, val_predictions, marker='.')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Val Predictions')
    ax2.axis('equal')
    ax2.axis('square')
    _ = ax2.plot([-100, 100], [-100, 100])

    ax3.scatter(test_y, test_predictions, marker='.', color='g')
    ax3.set_xlabel('True Values')
    ax3.set_ylabel('Predictions')
    ax3.set_title('Test Predictions')
    ax3.axis('equal')
    ax3.axis('square')
    _ = ax3.plot([-100, 100], [-100, 100])
    plt.savefig(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}PREDICT{os.sep}{MODEL_NAME}_{NOW}.png')

    # plt\.show\(\)
    plt.close()

    # In[ ]:

    _, ax = plt.subplots()
    ax.scatter(x=range(0, test_y.size), y=test_y,
               c='blue', label='Actual', alpha=0.3)
    ax.scatter(x=range(0, test_predictions.size),
               y=test_predictions, c='red', label='Predicted', alpha=0.3)

    plt.title('Actual and predicted values')
    plt.xlabel('Observations')
    plt.ylabel('mpg')
    plt.legend()
    # plt\.show\(\)
    plt.close()

    # In[ ]:

    diff = test_y.reshape(test_predictions.shape) - test_predictions
    plt.hist(diff, bins=40)
    plt.title('Histogram of prediction errors')
    plt.xlabel('MPG prediction error')
    plt.ylabel('Frequency')
    plt.close()

    # **Plot metrics and losses**

    # In[ ]:

    def smooth_curve(points, factor=0.75):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(
                    previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    # In[ ]:

    os.makedirs(f'reports{os.sep}TRIAL{TRIAL}{os.sep}LOSS', exist_ok=True)
    plt.plot(smooth_curve(history.history['loss']))
    plt.plot(smooth_curve(history.history['val_loss']))
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}LOSS{os.sep}{MODEL_NAME}_{NOW}.png')
    # plt\.show\(\)
    plt.close()

    # In[ ]:

    os.makedirs(f'reports{os.sep}TRIAL{TRIAL}{os.sep}R2', exist_ok=True)
    plt.plot(smooth_curve(history.history['r2_keras']))
    plt.plot(smooth_curve(history.history['val_r2_keras']))
    plt.title('R2')
    plt.ylabel('R2')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}R2{os.sep}{MODEL_NAME}_{NOW}.png')
    # plt\.show\(\)
    plt.close()

    # In[ ]:

    os.makedirs(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}Pearson', exist_ok=True)
    plt.plot(smooth_curve(history.history['pearson_r']))
    plt.plot(smooth_curve(history.history['val_pearson_r']))
    plt.title('Pearson')
    plt.ylabel('Pearson')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}Pearson{os.sep}{MODEL_NAME}_{NOW}.png')
    # plt\.show\(\)
    plt.close()

    # __Write csv__

    # In[ ]:

    hist_df = pd.DataFrame(history.history)

    os.makedirs(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}history', exist_ok=True)
    hist_csv_file = f'reports{os.sep}TRIAL{TRIAL}{os.sep}history/{TOPOLOGY}__{MODEL_NAME}_{NOW}.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # In[ ]:

    os.makedirs(
        f'reports{os.sep}TRIAL{TRIAL}{os.sep}predictions', exist_ok=True)

    pred_csv_file = f'reports{os.sep}TRIAL{TRIAL}{os.sep}predictions/{MODEL_NAME}_{NOW}.csv'

    with open(pred_csv_file, mode='w') as f:
        f.write("TRUE,PREDICTED\n")
        for y_true, y_pred in zip(test_y, test_predictions):
            f.write(f"{y_true},{y_pred[0]}\n")

    # In[ ]:

    if not os.path.isfile(f'reports{os.sep}TRIAL{TRIAL}{os.sep}report.csv'):
        with open(f'reports{os.sep}TRIAL{TRIAL}{os.sep}report.csv', 'a') as report:
            report.write('DATA,NOME,TRANSFORM,TIPO,TOPOLOGIA,DATASET,VAL_IS_TEST,'
                         'TARGET,AUGMENTATION,EPOCAS,LR,LOSSFUNC,'
                         'TRAIN_LOSS,TRAIN_MSE,TRAIN_R2,TRAIN_PEARSON,'
                         'VAL_LOSS,VAL_MSE,VAL_R2,VAL_PEARSON,'
                         'TEST_LOSS,TEST_MSE,TEST_R2,TEST_PEARSON,\n')
    with open(f'reports{os.sep}TRIAL{TRIAL}{os.sep}report.csv', 'a') as report:
        report.write(f'{NOW},{MODEL_NAME},{TRANSFORMATION},{TYPE},{TOPOLOGY},{DATASET},'
                     f'{USE_TEST_AS_VAL},{TARGET},{AUGMENT},{EPOCHS},{LR},{LOSSF},'
                     f'{train_loss},{train_mse},{train_r2},{train_p},'
                     f'{val_loss},{val_mse},{val_r2},{val_p},'
                     f'{test_loss},{test_mse},{test_r2},{test_p}\n')


if __name__ == "__main__":
    SHOW_INFO = '--showinfo' in sys.argv
    USE_TEST_AS_VAL = '--val==test' in sys.argv
    BALANCE_DATA = '--balance' in sys.argv

    COMMENTS = sys.argv[1]
    TARGET = sys.argv[2]
    AUGMENT = sys.argv[3]
    TOPOLOGY = sys.argv[4]
    TRIAL = sys.argv[5]
    SPLIT = int(sys.argv[6])
    NORMALIZATION_TYPE = sys.argv[7] if sys.argv[7] != "None" else None
    NORMALIZE_BY = sys.argv[8] if sys.argv[8] != "None" else None
    TRANSFORMATION = sys.argv[9] if sys.argv[9] in TRANSFORMATIONS else None
    
    if AUGMENT != 'default' and AUGMENT != 'simple':
        AUGMENT = None

    run(TRIAL=TRIAL,
        TOPOLOGY=TOPOLOGY,
        AUGMENT=AUGMENT,
        SHOW_INFO=SHOW_INFO,
        USE_TEST_AS_VAL=USE_TEST_AS_VAL,
        BALANCE_DATA=BALANCE_DATA,
        NORMALIZE_BY=NORMALIZE_BY,
        NORMALIZATION_TYPE=NORMALIZATION_TYPE)

    