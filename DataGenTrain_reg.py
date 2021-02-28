# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:32:58 2019

@author: mtabas0
"""
import numpy as np
# import keras
import tensorflow
from AugTrain_reg import *
import scipy.io as sio


from AugTrain_reg import AugTrain_reg


class DataGenTrain_reg(tensorflow.compat.v2.keras.utils.Sequence):  # tf.compat.v1.keras.utils.Sequence
    """Generates data for Keras"""

    def __init__(self, in_folder, out_folder, list_IDs, batch_size, dim, lr, n_lvl,
                 n_init_krn, num_out, minmax, ConvToFC):
        """Initialization"""
        self.dim = dim
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_init_krn = n_init_krn
        self.num_out = num_out
        self.minmax = minmax
        self.ConvToFC = ConvToFC

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) * 4 / self.batch_size))  # 3 is the number of augmentation

    def __getitem__(self, idx):
        """Generate one batch of data"""

        # Initialization
        X = np.empty([self.batch_size * 4,
                      self.dim[0], self.dim[1], self.dim[2], self.dim[3]])

        Y = np.empty([self.batch_size * 4,
                      self.num_out])

        batch = self.list_IDs[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Generate data
        c = 0
        for i, ID in enumerate(batch):

            # Load input and output

            raw_vol_in = np.array(sio.loadmat(self.in_folder[ID]).get("bModes"))
            raw_vol_in[np.isneginf(raw_vol_in)] = -151
            raw_vol_in = np.nan_to_num(raw_vol_in)

            tmp_vol_out = np.array(sio.loadmat(self.out_folder[ID]).get('regVars'))
            tmp_vol_out[tmp_vol_out < 1e-6] = 0
            print(tmp_vol_out[0])
            #            tmp_vol_out = np.nan_to_num(tmp_vol_out)

            tmp_vol_in = np.empty([self.dim[0], self.dim[1], self.dim[2]])

            for j in range(self.dim[2]):  # Extract input image in dim(128,128,99)

                tmp_vol_in[:, :, j] = raw_vol_in[:,
                                      j * self.dim[0]:self.dim[1] + j * self.dim[1]]  # selects all rows and shifts with

            # Call the data augmentation function
            Vols = AugTrain_reg(tmp_vol_in, tmp_vol_out, self.num_out, self.minmax)

            #            X_aug = Vols[0]
            #            Y_aug = Vols[1]


            X[i * c, ] = Vols[0][0] # original and augmented images in X
            X[i * c + 1, ] = Vols[0][1]
            X[i * c + 2, ] = Vols[0][2]
            X[i * c + 3, ] = Vols[0][3]


            Y[i * c, ] = Vols[1][0]
            Y[i * c + 1, ] = Vols[1][1]
            Y[i * c + 2, ] = Vols[1][2]
            Y[i * c + 3, ] = Vols[1][3]
            c = c + 4
            #print(Y)

        print("Shape in  datagentrain: " + str(type(X[1][1][1][45][0])))
        X = np.moveaxis(X,-2,1)
        print(X.shape)
        print(Y.shape)
        return X, Y
