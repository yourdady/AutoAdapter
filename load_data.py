''' 
@project AutoAdapter
@author Peng
@file load_data.py
@time 2018-08-19
'''
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import scipy.io as sio
import numpy as np
from PIL import Image
from dataSet import DataSet
MATFILE = 'data/usps_all.mat'
NUM_EXAMPLES = 11000
data = sio.loadmat(MATFILE)
image = data['data'][:, 4, 5].reshape(16, 16).T

class load_mnist():
    def __init__(self, validation_size = 5000, one_hot = False):
        """
        
        :param validation_size: number of validation samples, #train + #validation = 55000
        :param one_hot: bool, if true, get one-hot label.
        """
        mnist = read_data_sets("data/", one_hot=one_hot, validation_size=validation_size)
        self.dataset = mnist


def parse_data():
    tmp = [data['data'].reshape((256, 11000)).T.reshape((11000, 16, 16))]
    input_x = []
    for i in range(11000):
        img = Image.fromarray(tmp[0][i])


        img = np.array(img.resize((28, 28)))
        img = np.array(img)

        input_x.append(img.T)
    input_x = [input_x]
    for i in range(3):
        input_x = np.swapaxes(input_x, i, i + 1)
    input_y = []
    for i in range(11000):
        tmp = [0] * 10
        if (i % 10) != 9:
            tmp[i % 10 + 1] = 1
        else:
            tmp[0] = 1
        input_y.append(tmp)
    return input_x,input_y

class dataset():
    def __init__(self, train, test, validation):
        self.train = train
        self.test = test
        self.validation = validation
class load_usps(object):
    def __init__(self, images, labels, validation_size = 1000 ,test_size = 2000):
        self._images = images
        self._labels = labels
        self.train = DataSet(np.array(images[validation_size:(NUM_EXAMPLES - test_size)]),
                             np.array(labels[validation_size:(NUM_EXAMPLES - test_size)]))
        self.test = DataSet(np.array(images[NUM_EXAMPLES - test_size:]), np.array(labels[NUM_EXAMPLES - test_size:]))
        self.validation = DataSet(np.array(images[:validation_size]),
                                  np.array(labels[:validation_size]))
        self.dataset = dataset(self.train, self.test, self.validation)