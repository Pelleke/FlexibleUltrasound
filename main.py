# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 23:31:30 2019

@author: sakbar0
"""

import sys
import os.path
import numpy as np
#import keras
from keras.callbacks import *
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import pyplot

from DataGenTrain_reg import DataGenTrain_reg
from DataGenVal_reg import DataGenVal_reg
from model import Unet3D


image_dir = '/home/jannick/PycharmProjects/CNN/venv/Unet3D/Unet3D/images'
coeff_dir = '/home/jannick/PycharmProjects/CNN/venv/Unet3D/Unet3D/reg_coeff'
save_dir = '/home/jannick/PycharmProjects/CNN/venv/Unet3D/Unet3D/result/'


# define a function for sorting the samples that belong to tr/val/te sets
def id_sorting(id_list, image_dir, coeff_dir):

    id_dir_in = []
    id_dir_out = []
    cnt = 0

    for k in range(np.size(id_list)):

        # sbj_id = id_list[a]
        input_dir = image_dir + '/' + 'bMode_' + str(k+1) + '.mat'  # sort input data id
        id_dir_in.append(input_dir)
        out_dir = coeff_dir + '/' + 'regCoeff_' + str(k+1) + '.mat'  # sort ouput data id
        id_dir_out.append(out_dir)
        cnt = cnt + 1

    return id_dir_in, id_dir_out, cnt


def save_filters(layer):

    filters, biases = layer.get_weights()

    # normalise filters to enable plotting
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    fig = pyplot.figure()

    # select the first 10 filters
    n_filters, ix = 10, 1
    for k in range(n_filters):
        # get the filter
        f = filters[:, :, :, k, k]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1

    # show the figure
    # fig.show()

    my_path = '/home/jannick/PycharmProjects/CNN/venv/Unet3D/Unet3D/filters/'
    fig.savefig(my_path + 'filters_conv3d_1.png')

    return print('Filters saved on: ' + my_path)


# create the list of subjects that will be used for train and validation step
sbj_list = os.listdir('/home/jannick/PycharmProjects/CNN/venv/Unet3D/Unet3D/reg_coeff')

tr_sbj_id = sbj_list[0]  # list of train data
val_sbj_id = sbj_list[1]  # list of validation data
# ts_sbj_id = sbj_list[1]  # list of validation data
[id_dir_tr_in, id_dir_tr_out, cnt_tr] = id_sorting(tr_sbj_id, image_dir, coeff_dir )
[id_dir_val_in, id_dir_val_out, cnt_val] = id_sorting(val_sbj_id, image_dir, coeff_dir )

tr_id = np.arange(cnt_tr)
val_id = np.arange(cnt_val)

num_out = 1  # number of reg_coef
stp = 50   # number of frames in each image
img_sz = 128  # size of image

Coeff = np.empty([len(tr_sbj_id)+len(val_sbj_id), num_out])
i = 0
print(len(tr_sbj_id))
for l in range(2):

    coef = np.array(sio.loadmat(coeff_dir + '/' + 'regCoeff_' + str(l+1) + '.mat').get('regVars'))
    coef[:, 1][coef[:, 1] < 1e-5] = 0
    Coeff[i] = coef[:, 1]
    i = i + 1

# Extract total min and total max with formulate (x-total min/(total max-total min)) for Normalization
    # we can use these values to denormalize network output
totalmin = np.min(Coeff)
totalmax = np.max(Coeff)
minmax = [totalmin, totalmax]
print(minmax)


# make a directory for saving the weights
weight_dir = save_dir + 'Weights' + '/ValTeIDs'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)


prm_tr = {'dim': (img_sz, img_sz, stp, 1),
          'batch_size': 1,
          'lr': 1e-4,
          'in_folder': id_dir_tr_in,
          'out_folder': id_dir_tr_out,
          'list_IDs': tr_id,
          'n_init_krn': 32,  # can change number of initial kernel
          'n_lvl': 2,       # can change number of levels
          'num_out': num_out,
          'minmax':minmax,
          'ConvToFC': (32, 16, 8, 4, 2)}

prm_val = {'dim': (img_sz, img_sz, stp, 1),
          'batch_size': 1,
          'lr': 1e-4,
          'in_folder': id_dir_tr_in,
          'out_folder': id_dir_tr_out,
          'list_IDs': tr_id,
          'n_init_krn': 32,  # can change number of initial kernel
          'n_lvl': 3,        # can change number of levels
          'num_out': num_out,
          'minmax':minmax,
          'ConvToFC': (32, 16, 8, 4, 2)}  # we can change the number of kernels and layers in conv to fc layers

training_generator = DataGenTrain_reg(**prm_tr)  # generate the train data

train1 = training_generator.__getitem__(0)   # batch of input and output data a[0] is input images,a[1] is coeff
print("Shape in  main: " + str(type(train1[0][1][1][1][45][0])))
print(training_generator.__len__())   # length of each batch

validation_generator = DataGenVal_reg(**prm_val)  # generate validation data
val1 = validation_generator.__getitem__(0)

#Choose your model Unet3D or LRCN
model = Unet3D(**prm_tr)

weight_name = 'Unet3D_reg_nKrn' + str(prm_tr['n_init_krn']) + '_nLvl' + str(prm_tr['n_lvl'])

model_checkpoint = ModelCheckpoint(filepath=weight_dir+'/'+weight_name+'_epc'+"{epoch:03d}"+'_trloss'+"{loss:.5f}"+'_valloss'+"{val_loss:.5f}"+ ".hdf5",
                                                              monitor='val_loss',
                                                              save_best_only=True,
                                                              period=1)  # verbose=1,

ErlStp = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50,
                                                  verbose=0, mode='min', baseline=None, restore_best_weights=False)

hist = model.fit_generator(generator=training_generator,
                           steps_per_epoch=None, epochs=500, verbose=2,
                           validation_data=validation_generator, validation_steps=None,
                           callbacks=[model_checkpoint, ErlStp], shuffle=True)

# save trained model:
model.save("3DunetReg")
print("Model saved to disk")

# save filters in last block from first level in 3D unet as an image
save_filters(model.get_layer('conv3d_1'))

# reload model from disk
load_model("3DunetReg")

'''
# loss hist
tr_loss_history = hist.history["loss"]
val_loss_history = hist.history["val_loss"]

errFig = pyplot.figure()
pyplot.plot(tr_loss_history)
pyplot.plot(val_loss_history)
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper left')
errFig.show()

my_path_error = os.path.abspath('/Users/ramadankrasniqi/Documents/MasterIIW/thesiscode/Unet3D/errorHist/')
errFig.savefig(my_path_error + 'error_hist.png')
'''
# np.save(os.path.join(weight_dir, weight_name + '_tr_loss'), tr_loss_history)
# np.save(os.path.join(weight_dir, weight_name + '_val_loss'), val_loss_history)
