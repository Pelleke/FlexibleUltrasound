import numpy as np
import tensorflow
import scipy.io as sio
import matplotlib.pyplot as plt


class DataGenTest_reg(tensorflow.compat.v2.keras.utils.Sequence):  # tf.compat.v1.keras.utils.Sequence
    """Generates data for Keras"""

    def __init__(self, in_folder, out_folder, list_IDs, batch_size, dim, lr, n_lvl,
                 n_init_krn, num_out, minmax, ConvToFC):
        """Initialization"""
        self.dim = dim
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.batch_size = batch_size
        self.shuffle = False
        self.list_IDs = list_IDs
        self.n_init_krn = n_init_krn
        self.num_out = num_out
        self.minmax = minmax
        self.ConvToFC = ConvToFC

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))  # 3 is the number of augmentation

    def __getitem__(self, idx):
        """Generate one batch of data"""
        print("INDEX" + str(idx))
        # Initialization
        X = np.empty([self.batch_size,
                      self.dim[0], self.dim[1], self.dim[2], self.dim[3]], dtype=np.float64)

        Y = np.empty([self.batch_size,
                      self.num_out], dtype=np.float64)

        batch = self.list_IDs[idx * self.batch_size:(idx + 1) * self.batch_size]
        print(batch)
        k = 0
        # Generate data
        for i, ID in enumerate(batch):
            print("STEP: " + str(ID+1) + "/48")
            # Load input and output
            #print("Value" + str(np.array(sio.loadmat(self.out_folder[ID]).get('regVars'))[0, 1]))
            raw_vol_in = np.array(sio.loadmat(self.in_folder[ID]).get("bModes"))
            raw_vol_in[np.isneginf(raw_vol_in)] = 0
            raw_vol_in = np.nan_to_num(raw_vol_in)

            tmp_vol_out = np.array(sio.loadmat(self.out_folder[ID]).get('regVars'))
            #print(tmp_vol_out[0, 1])
            tmp_vol_in = np.empty([self.dim[0], self.dim[1], self.dim[2]])

            for j in range(self.dim[2]):  # Extract input image in dim(128,128,99)

                tmp_vol_in[:, :, j] = raw_vol_in[:,
                                      j*self.dim[0]:self.dim[1] + j * self.dim[1]]  # selects all rows and shifts with
            # Add channel dimension
            tmp_vol_in = np.expand_dims(tmp_vol_in, axis = -1)

            # Normalisation
            new_vol_out = (tmp_vol_out - self.minmax[0]) / (self.minmax[1] - self.minmax[0])
            new_vol_out[new_vol_out < 1e-5] = 0
            #new_vol_out = tmp_vol_out
            #print(new_vol_out[0,1])
            #print(self.minmax)
            X[i] = tmp_vol_in


            Y[i] = new_vol_out[:,1]


        #Reduceer initiele array size, korter door waarden 6.5 weg te gooien
        print(X.shape)
        #Y = Y[:,1]
        print("Y: " + str(Y))
        X = np.uint8(X)
        X = X/255
        print(np.average(X))
        #plt.imshow(X[0, :, :, 45, 0])
        #plt.show()
        return X #, Y, [None]