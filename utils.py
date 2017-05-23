import os
from scipy import misc
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.svm import SVC


def train(x, y, clf_path, kernel='poly'):
    """
    This function trains a SVM
    :param x: features
    :param y: labels
    :param clf_path: path where the classifier is saved
    :param kernel: default 'poly'
    :return: classifier
    """
    clf = SVC(kernel=kernel, gamma=0.1)
    clf.fit(x, y)
    joblib.dump(clf, clf_path)
    print('###Training done###')
    return clf


def get_path_label(folder='/Users/mohcine/Documents/Code/Scor/English/Img'):
    for dirpath, dirnames, filenames in os.walk(folder):
        if len(filenames) > 2 and 'Bmp' in dirpath:  # we choose 2 because of .DS_Store of the root folder
            for im in filenames:
                yield (str(dirpath) + '/' + str(im), int(dirpath.split('Sample')[1]))


def get_image(path, rgb2gray=True):
    if rgb2gray:
        return misc.imread(path, mode='L')
    else:
        return misc.imread(path)


def reshape_im(im, size=(28, 28, 3)):
    if len(im.shape) == 3:
        return misc.imresize(im, size)
    else:
        return misc.imresize(im, size[:2])


def normalize_im(im):
    return (im - np.mean(im)) / np.std(im)


def flatten_im(im, shape=784):
    return im.reshape((1, shape))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

