import data_loader
import graph_sub
import numpy as np
import tensorflow as tf
import argparse
import os
import warnings
import pdb
import psutil
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Few shot classification test")
parser.add_argument("-g", "--gpu", type = int, default = 0)
parser.add_argument("-tr", "--train", type = str, default = '1,2,3,4,5,6,7,8')
parser.add_argument("-te", "--test", type = str, default = '9')
parser.add_argument("-n", "--number", type = str, default = '1')
parser.add_argument("-m", "--model", type = str, default = 'HS_CNN')
parser.add_argument("-e", "--epoch", type = int, default = 100)
parser.add_argument("-l", "--learningrate", type = float, default =0.0001)

args = parser.parse_args()
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)

## Load data
with tf.variable_scope('Load_Data'):
    path = '/media/NAS/nas_187/sion/Dataset/EEG/numpy'
    train = list(map(int, args.train.split(',')))
    test = list(map(int, args.test.split(',')))
    if len(train) == 9:
        train.insert(0, '0')
    print("Train", train)
    print("Test", test)
    count = np.zeros(2, dtype = int)
    train_dict = {}
    val_dict = {}
    for subject_num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if subject_num in train:
            data, label = data_loader.data_load(path, subject_num, 'train', args.model)
            key = list(data.keys())
            for i in range(len(key)):
                sample = {}
                sample['data'] = data[key[i]][:, :875, :3]
                sample['label'] = label[key[i]]
                train_dict[key[i]] = sample
                count[0] += sample['data'].shape[0]
            data, label = data_loader.data_load(path, subject_num, 'test', args.model)
            key = list(data.keys())
            for i in range(len(key)):
                sample = {}
                sample['data'] = data[key[i]][:, :875, :3]
                sample['label'] = label[key[i]]
                if i == 0:
                    train_dict[key[i]] = sample
                    count[0] += sample['data'].shape[0]
                else:
                    val_dict[key[i]] = sample
                    count[1] += sample['data'].shape[0]

# pdb.set_trace()
    print('Model: ', args.model)
    print('Training  :', count[0], '개')
    print('Validation  :', count[1], '개')
#     print('Test  :', data_test.shape[0], '개')


## model information
with tf.variable_scope('Model_information'):
    if args.model == 'HS_CNN' or args.model == 'HS_CNN_IROS':
        kp = 0.8  ## dropout rate
    if args.model == 'EEGNet':
        kp = 0.25  ## dropout rate
    if args.model == 'ShallowNet':
        kp = 0.5  ## dropout rate
    if args.model == 'DeepconvNet':
        kp = 0.5  ## dropout rate
    lr = args.learningrate ## initial learning rate
    epochs = args.epoch ## Number of epochs
    mini_batch = 100 ## batch size
    n = 2 ## Number of layer
    kernel_list = []
    kernel_list = np.asarray(kernel_list)
    channel_list = []
    stride_list = []
    dense_list = [2]
    maxpool_list = [[1, 6, 1, 6, 1]]
    maxpool_list = np.asarray(maxpool_list)

## Graph build
with tf.variable_scope('Graph_build'):
    tf.set_random_seed(777)
    graph_sub.Build_graph(train_dict, val_dict, epochs, mini_batch, lr, str(test[0]), kp, channel_list,
                      kernel_list, stride_list, dense_list, maxpool_list, args.model) #data, data_val, data_test, label, label_val, label_test,


