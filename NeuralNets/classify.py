import tensorflow as tf
import sys
import pickle
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np

sys.path.append('../')
from NeuralNets import nntools
import utils

path_db = '/Users/mohcine/Documents/Code/Scor/Database/db.pickle'
model_path = '/Users/mohcine/Documents/Code/Scor/models/model-495000'
filename_train = '/Users/mohcine/Documents/Code/Scor/Database/englishlmg_train.tfrecords'
# Network parameters
nbr_hidden1 = 1000
nbr_hidden2 = 500
nbr_hidden3 = 250
nbr_hidden4 = 30
nbr_input = 28*28
batch_size = 64

db = pickle.load(open(path_db, 'rb'))

#  generation input for the autoencoder
train_im = [utils.flatten_im(utils.normalize_im(utils.reshape_im(utils.get_image(path)))) for path in db['train_path']]
train_lbl = db['train_label']


def inference(x):
    return nntools.encoder(x, nbr_input, nbr_hidden1, nbr_hidden2, nbr_hidden3, nbr_hidden4)

#  generating features for the training database
train_h = [nntools.ae_feed(data_in, model_path, inference, nbr_input) for data_in in train_im]

train_h_ = [el[0].reshape(30) for el in train_h]
clf_path = '/Users/mohcine/Documents/Code/Scor/clf.p'

clf = utils.train(train_h_, train_lbl, clf_path)