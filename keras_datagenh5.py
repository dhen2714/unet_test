"""
Keras based image data generator which yields data from hdf5 file.

Based on code from
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
See also
https://keras.io/api/utils/python_utils/#sequence-class
"""
import numpy as np
import keras
import h5py


class DataGeneratorH5(keras.utils.Sequence):
    """
    Data generator from HDF5 file.
    Assumes data is stored in correct format for input into the neural network.
    """

    def __init__(self, images_h5path, labels_h5path, batch_size=32, 
        shuffle=True, rescale=None):

        self.batch_size = batch_size

        with h5py.File(images_h5path, 'r') as f:
            self.list_IDs = list(f.keys())
        
        self.images_h5path = images_h5path
        self.labels_h5path = labels_h5path
        self.shuffle = shuffle
        self.rescale = rescale
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """
        Updates indices after each epoch, also called at very beginning.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        X = []
        y = []

        if self.rescale:
            rescale = self.rescale

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            with h5py.File(self.images_h5path, 'r') as f:
                X.append(f[ID][()])

            with h5py.File(self.labels_h5path, 'r') as f:
                y.append(f[ID][()])

        return rescale*np.array(X), rescale*np.array(y)


if __name__ == "__main__":
    h5path1 = 'warped.h5'
    h5path2 = 'notwarped.h5'
    dgen = DataGeneratorH5(h5path1, h5path2, batch_size=40)

    for i, lul in enumerate(dgen):
        print(lul[0].shape)
        print(lul[1].shape)
        if i == 10:
            break