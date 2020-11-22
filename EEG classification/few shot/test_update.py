#-*-coding: utf-8-*-
## conventional CNN

import data_loader
import data_loader_2a
import data_loader_HGD
import test_graph_update
import test_graph_att_update
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
parser.add_argument("-f", "--few", type = int, default = 1)
parser.add_argument("-d", "--data", type = str, default = "BCI4_2b")
parser.add_argument("-t", "--type", type = str, default = "")
parser.add_argument("-fine", "--finetuning", type = int, default= 0)

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
    load_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/Fewshot/'+str(args.few)+'/'+args.model+args.type+'/subject_' + str(test[0]) + '/'+args.model
    f = open(load_path + '/log.txt')
    lines = f.readlines()
    file_num = lines[-2].split(':')[1]
    file_num = file_num.split(',')[0]
    file_num = file_num.split(' ')[1]
    f.close()
    restore_path = load_path + '/best/model-' + str(int(file_num) - 1)
    save_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/Fewshot/'+str(args.few)+'/'+args.model+args.type+'/result/'+args.data+'_'+str(test[0])+'.txt'

    print("Test", test)
    count = 0
    data_dict = {}
    label_dict = {}
    for subject_num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if subject_num in test:
            if args.data == 'BCI4_2b':
                data__, label__ = data_loader.data_load(path, subject_num, 'train', args.model)
            if args.data == 'BCI4_2a':
                data__, label__ = data_loader_2a.data_load(path, subject_num, 'train', args.model)
            if args.data == 'HGD':
                data__, label__ = data_loader_HGD.data_load(path, subject_num, 'train', args.model)
            key = list(data__.keys())
            for i in range(len(key)):
                data_ = data__[key[i]]
                label_ = label__[key[i]]
                data_dict[key[i]] = data_[::, :875, :3]
                label_dict[key[i]] = label_

                count += data_.shape[0]

            if args.data == 'BCI4_2b':
                data__, label__ = data_loader.data_load(path, subject_num, 'test', args.model)
            if args.data == 'BCI4_2a':
                data__, label__ = data_loader_2a.data_load(path, subject_num, 'test', args.model)
            if args.data == 'HGD':
                data__, label__ = data_loader_HGD.data_load(path, subject_num, 'test', args.model)
            key = list(data__.keys())
            for i in range(len(key)):
                data_ = data__[key[i]]
                label_ = label__[key[i]]
                data_dict[key[i]] = data_[::, :875, :3]
                label_dict[key[i]] = label_

                count += data_.shape[0]
    print('N-shot : ', args.few)
    print('subject: ', test[0])
    print('Model  :', args.model+args.type)
    print('Test   :', count, '개')
    print(lines[-2].split('\n')[0])

## model information
with tf.variable_scope('Model_information'):
    if args.model == 'HS_CNN':
        kp = 0.8 ## dropout rate
    if args.model == 'EEGNet':
        kp = 0.25 ## dropout rate
    if args.model == 'ShallowNet':
        kp = 0.5 ## dropout rate
    if args.model == 'DeepconvNet':
        kp = 0.5 ## dropout rate
    lr = 1e-4 ## initial learning rate
    epochs = 20 ## Number of epochs
    mini_batch = 50 ## batch size
    n = 2 ## Number of layer
    kernel_list = [[[45, 1], [1, 2]]*int(n/2), [[65, 1], [1, 2]]*int(n/2), [[85, 1], [1, 2]]*int(n/2)]
    kernel_list = np.asarray(kernel_list)
    channel_list = [10, 10]
    stride_list = [[1, 1]]*n
    dense_list = [2]
    maxpool_list = [[1, 6, 1, 6, 1]]
    maxpool_list = np.asarray(maxpool_list)

## Graph build
with tf.variable_scope('Graph_build'):
    tf.set_random_seed(777)
    type_list = ['_att', '_gat', '_gcn', '_dis', '_satt']
    if args.type in type_list:
        test_graph_att_update.Build_graph(data_dict, label_dict, epochs, mini_batch, lr, str(test[0]), kp, channel_list,
                                          kernel_list, stride_list, dense_list, maxpool_list, restore_path, save_path,
                                          args.model, args.few, args.finetuning, args.type, args.data)
    else:
        test_graph_update.Build_graph(data_dict, label_dict, epochs, mini_batch, lr, str(test[0]), kp, channel_list,
                          kernel_list, stride_list, dense_list, maxpool_list, restore_path, save_path, args.model, args.few, args.finetuning, args.data)

