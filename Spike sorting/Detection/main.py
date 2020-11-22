import data_loader
import graph_build
import tensorflow as tf
import numpy as np
import os
import argparse
import warnings
import pdb
import random

parser = argparse.ArgumentParser(description="Few shot classification test")
parser.add_argument("-g", "--gpu", type = int, default = 0)
parser.add_argument("-l", "--learningrate", type = float, default = 1e-4)
parser.add_argument("-b", "--batch", type = int, default = 1000)
parser.add_argument("-e", "--epoch", type = int, default = 10000)

args = parser.parse_args()
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)

data_path = '/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week'
save_path = '/media/sion/D/Data'
code_path = '/home/sion/Desktop/code/EEG/spikesorting/Detection'

if not os.path.isfile(os.path.join(data_path, 'npy/train.npy')):
    file_list = os.listdir(os.path.join(data_path, 'data_label/data'))
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for i in range(len(file_list)):
        signal, label = data_loader.Load_data(file_list[i])
        type, number = np.unique(np.argmax(label, axis = 1), return_counts = True)
        count = np.zeros(2, dtype = int)
        data = []
        for j in range(label.shape[1]):
            sample = []
            for k in range(label.shape[0]):
                if label[k, j] == 1.0:
                    sample.append(signal[k:k+1])
            if len(sample) != 0:
                data.append(np.concatenate(sample, axis = 0))
            else:
                pass # 비어있는 cluster가 있음

        train_data = []
        val_data = []
        test_data = []
        train_label = []
        val_label = []
        test_label = []
        for j in range(len(data)):
            label_c = np.zeros((data[j].shape[0], len(data)))
            label_c[:, j] = 1
            sample_list = list(range(data[j].shape[0]))
            idx = random.sample(sample_list, int(data[j].shape[0]*0.4))
            for k in range(len(idx)):
                sample_list.remove(idx[k])
            train_data.append(data[j][sample_list])
            train_label.append(label_c[sample_list])
            val_data.append(data[j][idx[:int(len(idx)*0.5)]])
            val_label.append(label_c[idx[:int(len(idx)*0.5)]])
            test_data.append(data[j][idx[int(len(idx)*0.5):]])
            test_label.append(label_c[idx[int(len(idx)*0.5):]])

        sample = {}
        sample['signal'] = train_data
        sample['label'] = train_label
        train_dict[file_list[i]] = sample
        sample = {}
        sample['signal'] = val_data
        sample['label'] = val_label
        val_dict[file_list[i]] = sample
        sample = {}
        sample['signal'] = test_data
        sample['label'] = test_label
        test_dict[file_list[i]] = sample

    np.save(os.path.join(data_path, 'npy/train.npy'), train_dict)
    np.save(os.path.join(data_path, 'npy/val.npy'), val_dict)
    np.save(os.path.join(data_path, 'npy/test.npy'), test_dict)

else:
    train_dict = np.load(os.path.join(data_path, 'npy/train.npy'), allow_pickle=True).item()
    val_dict = np.load(os.path.join(data_path, 'npy/val.npy'), allow_pickle=True).item()
    test_dict = np.load(os.path.join(data_path, 'npy/test.npy'), allow_pickle=True).item()

# Model information
kernel = [[16, 1], [8, 1], [4, 1], [2, 1]]
channel = [8, 32, 128, 256]
stride = [[1, 1], [1, 1], [1, 1], [1, 1]]
dense = [128, 64, 16, 2]
pooling = np.asarray([[100, 1, 1, 1, 1]]) # No pooling
lr = args.learningrate
kp = 0.7
epochs = args.epoch
mini_batch = args.batch

with tf.variable_scope('Graph_build'):
    tf.set_random_seed(777)
    graph_build.Build_graph(train_dict, val_dict, kernel, channel, stride, dense, pooling, lr, kp, epochs, mini_batch, save_path, code_path)
