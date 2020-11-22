import sys
sys.path.append('/home/sion/code')
import Code_total as ct
import numpy as np
import tensorflow as tf
import os
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

def cross_entropy(output, y, lrate):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-10)+(1-y)*tf.log(1-output+1e-10), axis=1))  ## cross entropy
    reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  ## regularizer
    loss += tf.reduce_sum(reg)
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

def CNN_layer(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = tf.nn.elu, \
                                  reuse=reuse_, kernel_initializer=tf.random_normal_initializer(stddev=0.0001, mean=0))
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.contrib.layers.xavier_initializer(),

        return output

def CNN_layer_linear(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = None, \
                                  reuse=reuse_, kernel_initializer=tf.random_normal_initializer(stddev=0.0001, mean=0))
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.contrib.layers.xavier_initializer(),

        return output

def build_model_DeepconvNet(input, kp):
    with tf.variable_scope("Block1"):
        layer = CNN_layer_linear(input, 25, [10, 1], [1, 1], False, 'conv1', 'valid')
        layer = CNN_layer_linear(layer, 25, [1, 3], [1, 1], False, 'Depthconv', 'valid')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm1')
        layer = tf.nn.elu(layer)
        # layer = tf.layers.max_pooling2d(layer, pool_size = [3, 1], strides = [3, 1], padding = 'valid',
        #                                          name = 'maxpool1')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout1')
    with tf.variable_scope("Block2"):
        layer = CNN_layer_linear(layer, 50, [10, 1], [1, 1], False, 'conv2', 'valid')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm2')
        layer = tf.nn.elu(layer)
        layer = tf.layers.max_pooling2d(layer, pool_size=[3, 1], strides=[3, 1], padding='valid',
                                        name='maxpool2')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout2')
    with tf.variable_scope("Block3"):
        layer = CNN_layer_linear(layer, 100, [10, 1], [1, 1], False, 'conv3', 'valid')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm3')
        layer = tf.nn.elu(layer)
        # layer = tf.layers.max_pooling2d(layer, pool_size=[3, 1], strides=[3, 1], padding='valid',
        #                                 name='maxpool3')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout3')
    with tf.variable_scope("Block4"):
        layer = CNN_layer_linear(layer, 200, [10, 1], [1, 1], False, 'conv4', 'valid')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm4')
        layer = tf.nn.elu(layer)
        layer = tf.layers.max_pooling2d(layer, pool_size=[3, 1], strides=[3, 1], padding='valid',
                                        name='maxpool4')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout4')
    layer = tf.layers.flatten(layer)
    layer = tf.layers.dense(layer, units=2, activation=tf.nn.softmax, name='Softmax')

    return layer

def build_model_ShallowNet(input, kp):
    layer = CNN_layer_linear(input, 40, [25, 1], [1, 1], False, 'conv', 'same')
    layer = CNN_layer_linear(layer, 40, [1, 3], [1, 1], False, 'Depthconv', 'valid')
    layer = tf.layers.batch_normalization(layer, name = 'BatchNorm')
    layer = tf.square(layer)
    layer = tf.layers.average_pooling2d(layer, pool_size = [75, 1], strides = [7, 1], padding = 'valid',
                                             name = 'avgpool')
    layer = tf.log(layer)
    layer = tf.layers.flatten(layer)
    layer = tf.layers.dropout(layer, rate = kp, name = 'dropout')
    layer = tf.layers.dense(layer, units=2, activation=tf.nn.softmax, name='Softmax')

    return layer

def build_model_EEGNet(input, kp):
    with tf.variable_scope('Block1'):
        layer = CNN_layer_linear(input, 8, [125, 1], [1, 1], False, 'conv', 'same')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm')
        layer = CNN_layer_linear(layer, 16, [1, 3], [1, 1], False, 'Depthconv', 'valid')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm-Depth')
        layer = tf.nn.elu(layer)
        layer = tf.layers.average_pooling2d(layer, pool_size = [4, 1], strides = [2, 1], padding = 'same',
                                                 name = 'avgpool')
        layer = tf.layers.dropout(layer, rate = kp, name = 'dropout')

    with tf.variable_scope('Block2'):
        layer = CNN_layer_linear(layer, 16, [16, 1], [1, 1], False, 'conv', 'same')
        layer = tf.layers.batch_normalization(layer, name = 'BatchNorm')
        layer = tf.nn.elu(layer)
        layer = tf.layers.average_pooling2d(layer, pool_size = [8, 1], strides = [4, 1], padding = 'same',
                                                 name = 'avgpool')
        layer = tf.layers.dropout(layer, rate = kp, name = 'dropout')

    layer = tf.layers.flatten(layer)
    layer = tf.layers.dense(layer, units=2, activation=tf.nn.softmax, name='Softmax')

    return layer

def build_model_HS_CNN(input, kp, cl, sl, dl, re, name = ''):
    # input = input * 10000000

    for i in range(3):
        input_ = input[::, ::, ::, i:i + 1]
    # j = 0
    # input1 = input[::, ::, ::, 0:1]
    # input2 = input[::, ::, ::, 1:2]
    # input3 = input[::, ::, ::, 2:3]
        with tf.variable_scope('Freq'+str(i)):
            layer1 = CNN_layer(input_, 10, [45, 1], [3, 1], reuse_ = re, name = name+'45_'+str(i), padding='SAME')
            layer2 = CNN_layer(input_, 10, [65, 1], [3, 1], reuse_ = re, name = name+'65_'+str(i), padding='SAME')
            layer3 = CNN_layer(input_, 10, [85, 1], [3, 1], reuse_ = re, name = name+'85_'+str(i), padding='SAME')

            layer1 = CNN_layer(layer1, 10, [1, 3], [1, 1], reuse_=re, name=name + 'c45_' +str(i))
            layer2 = CNN_layer(layer2, 10, [1, 3], [1, 1], reuse_=re, name=name + 'c65_' +str(i))
            layer3 = CNN_layer(layer3, 10, [1, 3], [1, 1], reuse_=re, name=name + 'c85_' +str(i))

            layer1 = tf.layers.max_pooling2d(layer1, pool_size = [6, 1], strides = [6, 1], padding = 'VALID',
                                             name = 'max1-'+name+str(i))
            layer2 = tf.layers.max_pooling2d(layer2, pool_size=[6, 1], strides=[6, 1], padding='VALID',
                                             name='max2-' + name +str(i))
            layer3 = tf.layers.max_pooling2d(layer3, pool_size=[6, 1], strides=[6, 1], padding='VALID',
                                             name='max3-' + name +str(i))

            layer1 = tf.layers.flatten(layer1)
            layer2 = tf.layers.flatten(layer2)

            layer3 = tf.layers.flatten(layer3)

            if i == 0:
                layer = tf.concat([layer1, layer2, layer3], axis = -1)

            else:
                layer = tf.concat([layer, layer1, layer2, layer3], axis = -1)

    for i in range(len(dl) - 1):
        layer = tf.layers.dense(layer, units=dl[i], activation=tf.nn.elu,
                                name='FC_layer-' + name + str(int(i)))
        layer = tf.layers.dropout(layer, rate=kp, name='Dropout-' + name + str(int(i / 2)))

    layer = tf.layers.dense(layer, units=dl[-1], activation=tf.nn.softmax, name='Softmax-' + name)

    return layer

def Build_graph(data_dict, label_dict, epochs, mini_batch, lr, subject_num, kp, cl, kl, sl, dl, ml, restore_path, save_path, model, finetuning, dataset_name):
    with tf.variable_scope('Placeholder'):
        if model == "HS_CNN" or model == 'HS_CNN_IROS':
            x = tf.placeholder(tf.float32, shape=[None, 875, 3, 3])
        else:
            x = tf.placeholder(tf.float32, shape=[None, 875, 3, 1])
        y = tf.placeholder(tf.float32, shape = [None, 2])
        keep_prob = tf.placeholder(tf.float32)
        lrate = tf.placeholder(tf.float64)

    with tf.variable_scope('Model'):
        if model == 'HS_CNN' or model == 'HS_CNN_IROS':
            output = build_model_HS_CNN(x, keep_prob, cl, sl, dl, re = False, name ='')
        if model == 'EEGNet':
            output = build_model_EEGNet(x, keep_prob)
        if model == 'ShallowNet':
            output = build_model_ShallowNet(x, keep_prob)
        if model == 'DeepconvNet':
            output = build_model_DeepconvNet(x, keep_prob)


    with tf.variable_scope('Loss'):
        loss, train_step = cross_entropy(output, y, lrate)

    with tf.variable_scope('Accuracy'):
        prediction, answer, correct, accuracy = ct.class_accuracy(output, y)

    ## Open session for Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess, restore_path)
        if not os.path.exists(save_path.split('result')[0] + '/result'):
            os.makedirs(save_path.split('result')[0] + '/result')
        f = open(save_path, 'w')
        dataset = {}
        # for i in range(10):
        #     print('Loading...: ', i)
        #     if i == 0:
        #         data_dict_ = data_dict = np.load('/media/sion/F/dataset/EEG/numpy/data/BCI4_2b_meta/test_'
        #                                          +str(subject_num)+'_1.npy', allow_pickle='True').item()
        #     else:
        #         data_dict_ = data_dict = np.load('/media/sion/F/dataset/EEG/numpy/data/BCI4_2b_meta/test_'
        #                                          + str(subject_num) + '_1_'+str(i+1)+'.npy', allow_pickle='True').item()
        #     dataset[str(i)] = data_dict_
        key = list(data_dict.keys())
        for i in range(len(key)):
            if i != len(key):
                data_test = data_dict[key[i]]
                label_test = label_dict[key[i]].astype(np.int32)
            if dataset_name == 'BCI4_2b':
                if os.path.isfile(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_sup_list.npy'):

                    sup_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_sup_list.npy')
                    que_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_que_list.npy')
                else:
                    sup_list = np.zeros((20, 10, 2), dtype=int)
                    que_list = np.zeros((data_test.shape[0] - 40, 10), dtype=int)
                    for rerun in range(10):
                        random_list = []
                        class_count = np.zeros(2, dtype=int)
                        while True:
                            ran_num = random.randint(0, data_test.shape[0] - 1)
                            if class_count[int(label_test[ran_num, 1])] < 20 and ran_num not in random_list:
                                sup_list[
                                    class_count[int(label_test[ran_num, 1])], rerun, label_test[ran_num, 1]] = ran_num
                                class_count[int(label_test[ran_num, 1])] += 1
                                random_list.append(ran_num)
                            if class_count[0] == 20 and class_count[1] == 20:
                                break
                        number = list(range(data_test.shape[0]))
                        for que_ran in range(len(random_list)):
                            number.remove(random_list[que_ran])
                        que_list[:, rerun] = number
                    np.save(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_sup_list.npy', sup_list)
                    np.save(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_que_list.npy', que_list)

            if dataset_name == 'BCI4_2a':
                if os.path.isfile(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_sup_list.npy'):
                    sup_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_sup_list.npy')
                    que_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_que_list.npy')

                else:
                    sup_list = np.zeros((20, 10, 2), dtype=int)
                    que_list = np.zeros((data_test.shape[0] - 40, 10), dtype=int)
                    for rerun in range(10):
                        random_list = []
                        class_count = np.zeros(2, dtype=int)
                        while True:
                            ran_num = random.randint(0, data_test.shape[0] - 1)
                            if class_count[int(label_test[ran_num, 1])] < 20 and ran_num not in random_list:
                                sup_list[
                                    class_count[int(label_test[ran_num, 1])], rerun, label_test[ran_num, 1]] = ran_num
                                class_count[int(label_test[ran_num, 1])] += 1
                                random_list.append(ran_num)
                            if class_count[0] == 20 and class_count[1] == 20:
                                break
                        number = list(range(data_test.shape[0]))
                        for que_ran in range(len(random_list)):
                            number.remove(random_list[que_ran])
                        que_list[:, rerun] = number
                    np.save(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_sup_list.npy', sup_list)
                    np.save(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_que_list.npy', que_list)

            if dataset_name == 'HGD':
                if os.path.isfile(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_sup_list.npy'):
                    sup_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_sup_list.npy')
                    que_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_que_list.npy')
                else:
                    sup_list = np.zeros((20, 10, 2), dtype=int)
                    que_list = np.zeros((data_test.shape[0] - 40, 10), dtype=int)
                    for rerun in range(10):
                        random_list = []
                        class_count = np.zeros(2, dtype=int)
                        while True:
                            ran_num = random.randint(0, data_test.shape[0] - 1)
                            if class_count[int(label_test[ran_num, 1])] < 20 and ran_num not in random_list:
                                sup_list[
                                    class_count[int(label_test[ran_num, 1])], rerun, label_test[ran_num, 1]] = ran_num
                                class_count[int(label_test[ran_num, 1])] += 1
                                random_list.append(ran_num)
                            if class_count[0] == 20 and class_count[1] == 20:
                                break
                        number = list(range(data_test.shape[0]))
                        for que_ran in range(len(random_list)):
                            number.remove(random_list[que_ran])
                        que_list[:, rerun] = number
                    np.save(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_sup_list.npy', sup_list)
                    np.save(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_que_list.npy', que_list)

            for re in range(10):
                if finetuning != 0:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, restore_path)
                    support_list = data_dict[key[i]][sup_list[:, 0, 0]]
                    support_list_ = data_dict[key[i]][sup_list[:, 0, 1]]
                    tr = np.concatenate((support_list, support_list_), axis=0)
                    tr_label = np.zeros((40, 2))
                    tr_label[:20, 0] = 1
                    tr_label[20:, 1] = 1
                    for fine in range(finetuning):
                        sess.run(train_step, feed_dict={x: tr, y: tr_label, keep_prob: kp, lrate: 0.0005})
                training_accuracy = 0
                training_loss = 0
                # pdb.set_trace()
                # data_test = dataset[str(re)][key[i]]['query'][:, 0, :, :, :3] * 1e+8
                # label_test = dataset[str(re)][key[i]]['label']

                data_test = data_dict[key[i]][que_list[:, re]]
                label_test = label_dict[key[i]][que_list[:, re]]

                run = int(data_test.shape[0] / mini_batch)
                if data_test.shape[0] % mini_batch != 0:
                    run += 1
                for j in range(run):
                    tr = data_test[j * mini_batch:(j + 1) * mini_batch]
                    tr_label = label_test[j * mini_batch:(j + 1) * mini_batch]
                    predic_value, train_accuracy, loss_print, train_prediction = \
                        sess.run([output, accuracy, loss, prediction], feed_dict={x: tr, y: tr_label, keep_prob: 1.0})
                    training_accuracy += train_accuracy * tr.shape[0]
                    training_loss += loss_print * tr.shape[0]

                training_accuracy /= data_test.shape[0]
                training_loss /= data_test.shape[0]

                file = key[i]
                print(file, 'Number: ', data_test.shape[0], 'Accuracy: ', training_accuracy, 'Loss: ', training_loss)

                f.write(
                    '%s, Number: %d, Accuracy: %f, Loss: %f\n' % (file, data_test.shape[0], training_accuracy, training_loss))
        f.close()
