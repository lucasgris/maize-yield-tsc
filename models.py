from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, BatchNormalization, LeakyReLU, ReLU,
                                     MaxPooling2D, GlobalAveragePooling2D, Input, Concatenate,
                                     Activation, TimeDistributed, Flatten, Conv1D, ConvLSTM2D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence


# ---------------------------------------------
# -------------- DAS models -------------------
# ---------------------------------------------


def CNN_DAS_smallkernel_leakyrelu(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=1)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu_heavy(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=32,
               kernel_size=3)(input_im)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(25, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dense(5, kernel_regularizer=regularizers.l1(0.01))(z)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])
                 

def CNN_DAS_leakyrelu_heavy2_pool(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=64,
               kernel_size=3)(input_im)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(80, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dense(30, kernel_regularizer=regularizers.l1(0.005))(z)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu_heavy3_pool2(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=80,
               kernel_size=3)(input_im)
    x = MaxPooling2D(pool_size=(3, 3), strides=None, padding="valid")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(80, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dense(50, kernel_regularizer=regularizers.l1(0.005))(z)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu2(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
#     x = LeakyReLU(alpha=0.1)(x)
    X = ReLU()(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(merged)
#     z = LeakyReLU(alpha=0.1)(z)
    z = ReLU()(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu3(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(z)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu4(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(10, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def CNN_DAS_leakyrelu_pool(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def MLP_DAS_leakyrelu(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Dense(7, kernel_regularizer=regularizers.l1(0.01))(input_im)
    x = LeakyReLU(alpha=0.1)(x)
    merged = Concatenate()([x, input_das])
    z = Dense(5, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])


def MLP_DAS_leakyrelu2(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Dense(7, kernel_regularizer=regularizers.l1(0.01))(input_im)
    x = LeakyReLU(alpha=0.1)(x)
    merged = Concatenate()([x, input_das])
    z = Dense(2, kernel_regularizer=regularizers.l1(0.1))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    output = Dense(1)(z)
    return Model([input_im, input_das], [output])

# ---------------------------------------------
# -------------- 2 output models -------------------
# ---------------------------------------------


def CNN_DAS_YieldTSC_smallkernel_leakyrelu(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=1)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)

    merged = Concatenate()([x, input_das])
    z = Dense(5, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(3, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    yield_part = Dense(2, kernel_regularizer=regularizers.l1(0.005))(yield_part)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    tsc_part = Dense(2, kernel_regularizer=regularizers.l1(0.005))(z)
    tsc_part = LeakyReLU(alpha=0.1)(tsc_part)
    output_tsc = Dense(1, name='TSC')(tsc_part)
    
    return Model([input_im, input_das], [output_yield, output_tsc])


def CNN_DAS_YieldTSC_leakyrelu(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    
    merged = Concatenate()([x, input_das])
    z = Dense(5, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(3, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    yield_part = Dense(2, kernel_regularizer=regularizers.l1(0.005))(yield_part)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    tsc_part = Dense(2, kernel_regularizer=regularizers.l1(0.005))(z)
    tsc_part = LeakyReLU(alpha=0.1)(tsc_part)
    output_tsc = Dense(1, name='TSC')(tsc_part)
    
    return Model([input_im, input_das], [output_yield, output_tsc])


def CNN_DAS_YieldTSC_leakyrelu_heavy(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=32,
               kernel_size=3)(input_im)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(32, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(10, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    yield_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(yield_part)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    tsc_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(z)
    tsc_part = LeakyReLU(alpha=0.1)(tsc_part)
    output_tsc = Dense(1, name='TSC')(tsc_part)
    
    return Model([input_im, input_das], [output_yield, output_tsc])

                 

def CNN_DAS_YieldTSC_leakyrelu_heavy2_pool(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=64,
               kernel_size=3)(input_im)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(40, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(30, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    yield_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(yield_part)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    tsc_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(z)
    tsc_part = LeakyReLU(alpha=0.1)(tsc_part)
    output_tsc = Dense(1, name='TSC')(tsc_part)
    
    return Model([input_im, input_das], [output_yield, output_tsc])



def CNN_DAS_YieldTSC_leakyrelu_heavy3_pool2(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=80,
               kernel_size=3)(input_im)
    x = MaxPooling2D(pool_size=(3, 3), strides=None, padding="valid")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(60, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(30, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    output_tsc = Dense(1, name='TSC')(z)
    
    return Model([input_im, input_das], [output_yield, output_tsc])



def CNN_DAS_YieldTSC_leakyrelu2(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
#     x = LeakyReLU(alpha=0.1)(x)
    X = ReLU()(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    z = Dense(8, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    output_tsc = Dense(1, name='TSC')(z)
    
    return Model([input_im, input_das], [output_yield, output_tsc])


def MLP_DAS_YieldTSC_leakyrelu(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Dense(7, kernel_regularizer=regularizers.l1(0.01))(input_im)
    x = LeakyReLU(alpha=0.1)(x)
    merged = Concatenate()([x, input_das])
    
    z = Dense(8, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    output_tsc = Dense(1, name='TSC')(z)
    
    return Model([input_im, input_das], [output_yield, output_tsc])


def CNN_DAS_YieldTSC_leakyrelu_pool(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Conv2D(filters=1,
               kernel_size=3)(input_im)    
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
    x = Flatten()(x)
    merged = Concatenate()([x, input_das])
    
    z = Dense(8, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    output_tsc = Dense(1, name='TSC')(z)
    
    return Model([input_im, input_das], [output_yield, output_tsc])

def MLP_DAS_YieldTSC_leakyrelu2(shape):
    input_im = Input(shape=shape)
    input_das = Input(shape=(1,))
    
    x = Dense(7, kernel_regularizer=regularizers.l1(0.01))(input_im)
    x = LeakyReLU(alpha=0.1)(x)
    merged = Concatenate()([x, input_das])
    
    
    z = Dense(8, kernel_regularizer=regularizers.l1(0.01))(merged)
    z = LeakyReLU(alpha=0.1)(z)
    
    yield_part = Dense(5, kernel_regularizer=regularizers.l1(0.005))(z)
    yield_part = LeakyReLU(alpha=0.1)(yield_part)
    output_yield = Dense(1, name='Yield')(yield_part)
    
    output_tsc = Dense(1, name='TSC')(z)
    
    return Model([input_im, input_das], [output_yield, output_tsc])