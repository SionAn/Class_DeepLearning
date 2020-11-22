## conventional CNN

import data_loader
import data_loader_2a
import data_loader_HGD
import test_graph_sub
import numpy as np
import tensorflow as tf
import argparse
import os
import warnings
import pdb

parser = argparse.ArgumentParser(description="Few shot classification test")
parser.add_argument("-g", "--gpu", type = int, default = 0)
parser.add_argument("-tr", "--train", type = str, default = '1,2,3,4,5,6,7,8')
parser.add_argument("-te", "--test", type = str, default = '9')
parser.add_argument("-n", "--number", type = str, default = '1')
parser.add_argument("-m", "--model", type = str, default = 'HS_CNN')
parser.add_argument("-d", "--data", type = str, default = 'BCI4_2b')
parser.add_argument("-fine", "--finetuning", type = int, default = 0)

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
    load_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/supervised/'+args.model+'/subject_' + str(test[0]) + '/'+ args.model
    f = open(load_path + '/log.txt')
    lines = f.readlines()
    file_num = lines[-2].split(':')[1]
    file_num = file_num.split(',')[0]
    file_num = file_num.split(' ')[1]
    f.close()
    restore_path = load_path + '/model-' + str(int(file_num) - 1)
    save_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/supervised/'+args.model+'/result/'+args.data+'_'+str(test[0])+'.txt'

    if args.data == 'BCI4_2a':
        test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if args.data == 'HGD':
        test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    print("Test", test)
    print("Model: ", args.model)
    count = 0
    data_dict = {}
    label_dict = {}
    for subject_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        if subject_num in test:
            if args.data == 'BCI4_2a':
               data, label = data_loader_2a.data_load(path, subject_num, 'train', args.model)
            if args.data == 'BCI4_2b':
               data, label = data_loader.data_load(path, subject_num, 'train', args.model)
            if args.data == 'HGD':
               data, label = data_loader_HGD.data_load(path, subject_num, 'train', args.model)
            key = list(data.keys())
            for i in range(len(key)):
                data_dict[key[i]] = data[key[i]][:, :875, :3]
                label_dict[key[i]] = label[key[i]]
                count += data[key[i]][:, :875, :3].shape[0]
            if args.data == 'BCI4_2a':
               data, label = data_loader_2a.data_load(path, subject_num, 'test', args.model)
            if args.data == 'BCI4_2b':
               data, label = data_loader.data_load(path, subject_num, 'test', args.model)
            if args.data == 'HGD':
               data, label = data_loader_HGD.data_load(path, subject_num, 'test', args.model)
            key = list(data.keys())
            for i in range(len(key)):
                data_dict[key[i]] = data[key[i]][:, :875, :3]
                label_dict[key[i]] = label[key[i]]
                count += data[key[i]][:, :875, :3].shape[0]

    print('Test  :', count, '개')
    print(lines[-2].split('\n')[0])

## model information
with tf.variable_scope('Model_information'):
    if args.model == 'HS_CNN' or args.model == 'HS_CNN_IROS':
        kp = 0.8 ## dropout rate
    if args.model == 'EEGNet':
        kp = 0.25 ## dropout rate
    if args.model == 'ShallowNet':
        kp = 0.5 ## dropout rate
    if args.model == 'DeepconvNet':
        kp = 0.5 ## dropout rate
    lr = 1e-4 ## initial learning rate
    epochs = 20 ## Number of epochs
    mini_batch = 10 ## batch size
    n = 2 ## Number of layer
    kernel_list = [[[45, 1], [1, 2]]*int(n/2), [[65, 1], [1, 2]]*int(n/2), [[85, 1], [1, 2]]*int(n/2)]
    kernel_list = np.asarray(kernel_list)
    channel_list = [10, 10]
    stride_list = [[1, 1]]*n
    dense_list = [2]
    maxpool_list = [[1, 6, 1, 6, 1]]
    maxpool_list = np.asarray(maxpool_list)

# count = 0
# while True:
#     if label[count, 1] == 1:
#         break
#
#     count += 1
#
# data = np.concatenate((data[:10], data[count:count+10]), axis = 0)
# label = np.concatenate((label[:10], label[count:count+10]), axis = 0)

## Graph build
with tf.variable_scope('Graph_build'):
    tf.set_random_seed(777)
    test_graph_sub.Build_graph(data_dict, label_dict, epochs, mini_batch, lr, str(test[0]), kp, channel_list,
                      kernel_list, stride_list, dense_list, maxpool_list, restore_path, save_path, args.model, args.finetuning, args.data)


