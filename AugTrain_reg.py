# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:53:13 2019

@author: mtabas0
"""

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import rotate, shift
from skimage import data, io
from skimage.transform import resize
from skimage import transform as tr
from numpy import empty
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def AugTrain_reg(in_vol_in, in_vol_out, num_out, minmax):
    new_vol_in = np.empty([in_vol_in.shape[0], in_vol_in.shape[1], in_vol_in.shape[2], 1])
    new_vol_out = np.zeros((num_out, 1))
    print(in_vol_in.shape)
    sh = new_vol_in.shape
    Vols = [[], []]
    datagen = ImageDataGenerator()
    r_list = [-0.5, 0.25, 0.11]
    flip_list = [False, False, True]

    for i in range(len(r_list)):
        # prepare i different transformations
        r = r_list[i]
        flip = flip_list[i]
        # print("r= ", r)
        imageData = np.empty([sh[0], sh[1], sh[2]])
        image = np.zeros([sh[0], sh[1], 1])
        #print("Shape before augmentation: " + str(type(new_vol_in[1][1][45][0])))
        for j in range(sh[2]):
            # apply transformation on image
            image1 = np.expand_dims(in_vol_in[:, :, j], 2)

            #print(image1)
            image = datagen.apply_transform(image1, transform_parameters={'flip_vertical': flip,
                                                                          # vertical flip results in upside down image
                                                                          'tx': r * 20,  # vertical translation
                                                                          'ty': r * 10})  # horizontal translation

            # Do preprocess on input images and change gray level of images to [0 255] and eliminate nan and -inf values from image pixles value and active below process
            image = np.squeeze(image, axis=2)

            image = np.uint8(image)
            #print(image)
            image[image < 0] = 0
            image[image > 255] = 255
            image = image/255
            #print("Val " + str(image[64,64]))
            #plt.imshow(image[:, :])
            #plt.show()
            #print("IMAGE " + str(image))
            #print("VAL: " + str(image[1][1]))
            imageData[:, :, j] = image  # (128,128,99)
            #plt.imshow(imageData[:, :, 45])
            #plt.show()
        imageData = imageData.reshape(sh[0], sh[1], sh[2], 1)
        Vols[0].append(imageData)

    # Add orgilal image in Vols[0]/ After preprocess of input images , active below process

    in_vol_in = np.uint8(in_vol_in)
    #in_vol_in = in_vol_in / 255
    in_vol_in[in_vol_in < 0] = 0
    in_vol_in[in_vol_in > 255] = 255
    in_vol_in = in_vol_in.reshape(sh[0], sh[1], sh[2], 1)
    Vols[0].append(in_vol_in)
    #print("Shape after augmentation: " + str(type(in_vol_in[1][1][45][0])))
    # Normalized coeff with total min and total max of all coefficients
    #print("voor"+ str(in_vol_out))
    new_vol_out= (in_vol_out - minmax[0]) / (minmax[1] - minmax[0])
    #print("na"+ str(new_vol_out))
    Vols[1].append(new_vol_out[:, 1])
    Vols[1].append(new_vol_out[:, 1])
    Vols[1].append(new_vol_out[:, 1])
    Vols[1].append(new_vol_out[:, 1])

    #    Vols.append(new_vol_in)
    #    Vols.append(new_vol_out)

    return Vols
