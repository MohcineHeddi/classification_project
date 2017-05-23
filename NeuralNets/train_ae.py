import tensorflow as tf
import sys

sys.path.append('../')
from NeuralNets import nntools


filename_train = '/home/mohcine/work/scor/Database/englishlmg_train.tfrecords'
num_epochs = 20000
batch_size = 64
learning_rate = 0.01
device_name = '/gpu:0'
# Network parameters
nbr_hidden1 = 1000
nbr_hidden2 = 500
nbr_hidden3 = 250
nbr_hidden4 = 30
nbr_input = 28*28


def inference(x):
    return nntools.decoder(nntools.encoder(x, nbr_input, nbr_hidden1, nbr_hidden2, nbr_hidden3, nbr_hidden4), nbr_input,
                           nbr_hidden1, nbr_hidden2, nbr_hidden3, nbr_hidden4)

model_path = '/home/mohcine/work/scor/models'
summary_path = '/home/mohcine/work/scor/summary'
nntools.ae_train(model_path, inference, filename_train, batch_size, num_epochs, learning_rate, device_name,
                 sum_path=summary_path)
