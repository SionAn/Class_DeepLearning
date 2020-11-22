import data_loader
import graph_build
import tensorflow as tf
import numpy as np
import os
import argparse
import warnings
import pdb

parser = argparse.ArgumentParser(description="Data representation")
parser.add_argument("-g", "--gpu", type = int, default = 0)
parser.add_argument("-l", "--learningrate", type = float, default = 1e-3)
parser.add_argument("-b", "--batch", type = int, default = 1000)
parser.add_argument("-e", "--epoch", type = int, default = 10000)
parser.add_argument("-n", "--dict_number", type = int, default = 1000)

args = parser.parse_args()
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)

data_path = '/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week'
save_path = '/media/sion/D/Data'
code_path = '/home/sion/Desktop/code/EEG/spikesorting/Representation'

train_dict = np.load(os.path.join(data_path, 'npy/train.npy'), allow_pickle=True).item()
val_dict = np.load(os.path.join(data_path, 'npy/val.npy'), allow_pickle=True).item()
train_idx = np.load(os.path.join(data_path, 'npy/train_idx.npy'), allow_pickle=True).item()
val_idx = np.load(os.path.join(data_path, 'npy/val_idx.npy'), allow_pickle=True).item()

# Model information
kernel = [[16, 1], [8, 1], [4, 1], [2, 1]]
channel = [8, 32, 128, 256]
stride = [[1, 1], [1, 1], [1, 1], [1, 1]]
dense = [128, 64, 16]
pooling = np.asarray([[100, 1, 1, 1, 1]]) # No pooling
lr = args.learningrate
kp = 0.7
epochs = args.epoch
mini_batch = args.batch
dict_num = args.dict_number

with tf.variable_scope('Graph_build'):
    tf.set_random_seed(777)
    graph_build.Build_graph(train_dict, val_dict, train_idx, val_idx, kernel, channel, stride, dense, pooling, lr, kp, epochs, mini_batch,
                            save_path, code_path, dict_num)