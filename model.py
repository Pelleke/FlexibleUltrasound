import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, PReLU, BatchNormalization, Dropout,\
                                    Flatten, Reshape
from tensorflow.keras.layers import Input, add, concatenate, Dense
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
TimeDistributed = tf.keras.layers.TimeDistributed
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation
MaxPooling2D = tf.keras.layers.MaxPooling2D
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
regularizers = tf.keras.regularizers
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:03:53 2019

@author: Somayeh
"""



def add_default_block(model, kernel_filters, init, reg_lambda):
    # conv
    model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                     kernel_initializer=init, kernel_regularizer=regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # conv
    model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                     kernel_initializer=init, kernel_regularizer=regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # max pool
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    return model


def LRCN(**prm):

    inputs = prm['dim']        # dim = (128, 128, 99, 1)
    n_krn = prm['n_init_krn']  # n_init_krn = 32

    initialiser = 'glorot_uniform'
    reg_lambda = 0.001

    model = tf.keras.Sequential()

    # first (non-default) block
    model.add(TimeDistributed(Conv2D(n_krn, (7, 7), strides=(1, 1), padding='same',
                                     kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(reg_lambda)),
                                     input_shape=(50, 128, 128, 1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(
        TimeDistributed(Conv2D(n_krn, (3, 3), kernel_initializer=initialiser,
                               kernel_regularizer=regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) blocks
    # model = add_default_block(model, 32, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 64, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

    # LSTM output head
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(1, activation=None))
    opt = tf.keras.optimizers.Adam(lr=1e-4, clipnorm=1.)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

    #model.build(inputs)
    model.summary()
    return model

def Unet3D(**prm):
    inputs = Input(prm['dim'])  # dim = (128, 128, 99, 1)
    n_krn = prm['n_init_krn']  # n_init_krn = 32

    Denc = {}  # encoding dictionary
    for i in range(prm['n_lvl']):  # n_lvl = 4, so loop iterates over 0, 1, 2, 3

        if i == 0:
            inp = inputs

        # Part between {} is replace by .format()
        # Conv3D: first parameter is the amount of filters, e.g. in Cnv1: filters = 2^0 * 32 = 32
        Denc['Cnv{0}'.format(i + 1)] = Conv3D((2 ** i) * n_krn, 3, padding='same', data_format="channels_last",
                                              strides=1)(inp)
        Denc['BN{0}'.format(i + 1)] = BatchNormalization()(Denc['Cnv' + str(i + 1)])
        Denc['Act{0}'.format(i + 1)] = Activation('relu')(Denc['BN' + str(i + 1)])
        # in  Cnv2 the amount of filters is increased to 64
        Denc['Cnv{0}'.format(i + 1)] = Conv3D((2 ** (i + 1)) * n_krn, 3, padding='same', strides=1)(Denc['Act' + str(i + 1)])
        Denc['BN{0}'.format(i + 1)] = BatchNormalization()(Denc['Cnv' + str(i + 1)])
        Denc['Act{0}'.format(i + 1)] = Activation('relu')(Denc['BN' + str(i + 1)])
        # Denc['Drp{0}'.format(i + 1)] = Dropout(0.25)(Denc['Act' + str(i + 1)])
        Denc['Pool{0}'.format(i + 1)] = MaxPooling3D(pool_size=(2, 2, 1))(Denc['Act' + str(i + 1)])

        inp = Denc['Pool' + str(i + 1)]  # set inputs of nex level to shape of last layer in previous level

    #sz_img = prm['dim'][0] / (2 ** (prm['n_lvl']-1))  # (n_lvl = 4) sz_img = 128 / 2^4 = 8
    #sz_img = int(sz_img)

    #mapn = int((2 ** prm['n_lvl']) * n_krn)  # mapn =  2^4 * 32 = 512

    # Code to make the final layers of the network Fully Convolutional Layers
    #Fltn1 = Flatten()(Denc['Act' + str(prm['n_lvl'])])  # select final level in Denc Library, pool -> act
    #reshp1 = Reshape((int(sz_img), int(sz_img), mapn * 99, 1))(Fltn1)  # reshape so convolution can be computed

    CnvFC1 = Conv3D(prm['ConvToFC'][0], 5, activation='relu', strides=(1, 1, 1), padding='valid',
                    data_format="channels_last")(Denc['Act' + str(prm['n_lvl'])])

    Drp1 = Dropout(0.25)(CnvFC1)

    CnvFC2 = Conv3D(prm['ConvToFC'][1], 3, activation='relu', strides=(1, 1, 1), padding='valid',
                    data_format="channels_last")(Drp1)
    Drp2 = Dropout(0.25)(CnvFC2)

    CnvFC3 = Conv3D(prm['ConvToFC'][2], 3, activation='relu', strides=1, padding='valid', data_format="channels_last")(Drp2)

    Drp3 = Dropout(0.25)(CnvFC3)

    CnvFC4 = Conv3D(prm['ConvToFC'][3], 3, activation='relu', strides=1, padding='valid')(Drp3)

    Drp4 = Dropout(0.25)(CnvFC4)

    Fltn = Flatten()(Drp4)

    predictions = Dense(units=prm['num_out'], activation='relu')(Fltn)  # if no negative regression values change to ReLuange to ReLu

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(lr=prm['lr']), loss='mean_squared_error', metrics=['mae'])

    # tf.keras.utils.plot_model(model, to_file='3DRegressionModel.png',show_shapes=True, show_layer_names=True, rankdir='LR')

    model.summary()

    return model