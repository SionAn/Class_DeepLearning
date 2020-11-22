import sys
sys.path.append('/home/sion/code')
import data_loader
import Code_total as ct
import numpy as np
import tensorflow as tf
import os
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pdb
from MulticoreTSNE import MulticoreTSNE as TSNE

def next_batch(num, data1, data2, labels1):
    with tf.variable_scope('Suffle'):
        idx = np.arange(0, data1.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle1 = [data1[i] for i in idx]
        data_shuffle2 = [data2[i] for i in idx]
        labels_shuffle1 = [labels1[i] for i in idx]

    return np.asarray(data_shuffle1), np.asarray(data_shuffle2), np.asarray(labels_shuffle1)

def cross_entropy(output, y, lrate):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-10)+(1-y)*tf.log(1-output+1e-10), axis=1))  ## cross entropy
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

def Build_graph(data_dict, label_dict, epochs, mini_batch, lr, subject_num, kp, cl, kl, sl, dl, ml, restore_path, save_path, model, few, finetuning, type, data):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        try:
            saver = tf.train.import_meta_graph(restore_path.split('.')[0] + '.meta')
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, restore_path)
        except:
            saver = tf.train.import_meta_graph(restore_path.split('/best')[0]+restore_path.split('/best')[1].split('.')[0]+'.meta')
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, restore_path.split('/best')[0] + restore_path.split('/best')[1])

        x = tf.get_default_graph().get_tensor_by_name("Graph_build/Graph_build/Placeholder/Placeholder:0")
        x_ = tf.get_default_graph().get_tensor_by_name("Graph_build/Graph_build/Placeholder/Placeholder_1:0")
        y = tf.get_default_graph().get_tensor_by_name("Graph_build/Graph_build/Placeholder/Placeholder_2:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("Graph_build/Graph_build/Placeholder/Placeholder_3:0")
        output = tf.get_default_graph().get_tensor_by_name("Graph_build/Graph_build/New_Model/Relation/concat_2:0")

        with tf.variable_scope('Fine'):
            learningrate = tf.placeholder(tf.float64)
            loss, train_step = cross_entropy(output, y, learningrate)

        with tf.variable_scope('Result'):
            prediction, answer, correct, accuracy = ct.class_accuracy(output, y)

        if not os.path.exists(save_path.split('result')[0] + '/result'):
            os.makedirs(save_path.split('result')[0] + '/result')

        f = open(save_path, 'w')
        key = list(data_dict.keys())
        print(key)
        for i in range(len(key)):
            for re in range(10):
                training_accuracy = 0
                training_loss = 0
                if i != len(key):
                    data_test = data_dict[key[i]]
                    label_test = label_dict[key[i]]

                sup = np.zeros(
                    (data_test.shape[0] - 40, 20, data_test.shape[1], data_test.shape[2], data_test.shape[3], 2))
                que = np.zeros(
                    (data_test.shape[0] - 40, 1, data_test.shape[1], data_test.shape[2], data_test.shape[3]))

                if data == 'BCI4_2b':
                    sup_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_sup_list.npy')
                    que_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20/'
                        + str(key[i][:-4]) + '_que_list.npy')
                if data == 'BCI4_2a':
                    sup_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_sup_list.npy')
                    que_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_2a/'
                        + str(key[i][:-4]) + '_que_list.npy')
                if data == 'HGD':
                    sup_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_sup_list.npy')
                    que_list = np.load(
                        '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/support_list/result_RN_20_HGD/'
                        + str(key[i][:-4]) + '_que_list.npy')
                sup[0, :, :, :, :, 0] = data_test[sup_list[:, re, 0]]
                sup[0, :, :, :, :, 1] = data_test[sup_list[:, re, 1]]
                que[:, 0] = data_test[que_list[:, re]]
                label = label_test[que_list[:, re]]
                sup = sup[:, :few]
                que = que[:, :few]
                for j in range(1, que.shape[0]):
                    sup[j] = sup[0]
                sup, que, label = next_batch(sup.shape[0], sup, que, label)

                tr = np.zeros((2*few, few, sup.shape[2], sup.shape[3], sup.shape[4], 2))
                for j in range(2*few):
                    tr[j] = sup[0:1]
                tr_label = np.zeros((2 * few, 2))
                for j in range(few):
                    if j == 0:
                        tr_ = sup[0:1, j:j+1, :, :, :, 0]
                    else:
                        tr_ = np.concatenate((tr_, sup[0:1, j:j+1, :, :, :, 0]), axis = 0)
                    tr_label[j, 0] = 1

                for j in range(few):
                    tr_ = np.concatenate((tr_, sup[0:1, j:j+1, :, :, :, 1]), axis = 0)
                    tr_label[few+j, 1] = 1

                fine_tuning_number = finetuning
                if fine_tuning_number != 0:
                    try:
                        sess.run(tf.global_variables_initializer())
                        saver.restore(sess, restore_path)
                    except:
                        sess.run(tf.global_variables_initializer())
                        saver.restore(sess, restore_path.split('/best')[0] + restore_path.split('/best')[1])

                for fine in range(fine_tuning_number):
                    sess.run(train_step, feed_dict={x: tr, x_: tr_, y: tr_label, keep_prob: kp, learningrate: lr/10})

                run = int(que.shape[0] / mini_batch)
                if que.shape[0] % mini_batch != 0:
                    run += 1

                count = 0
                for j in range(run):
                    tr = sup[j * mini_batch:(j + 1) * mini_batch]
                    tr_ = que[j * mini_batch:(j + 1) * mini_batch]
                    tr_label = label[j * mini_batch:(j + 1) * mini_batch]
                    predic_value, train_accuracy, loss_print, train_prediction = \
                        sess.run([output, accuracy, loss, prediction],
                                 feed_dict={x: tr, x_: tr_, y: tr_label, keep_prob: 1.0})
                    training_accuracy += train_accuracy * tr.shape[0]
                    training_loss += loss_print * tr.shape[0]
                    count += tr.shape[0]

                training_accuracy /= count
                training_loss /= count

                if i != len(key):
                    file = key[i]
                else:
                    file = 'All data'

                print(file, 'Number: ', count, 'Accuracy: ', training_accuracy, 'Loss: ', training_loss)

                f.write(
                    '%s, Number: %d, Accuracy: %f, Loss: %f\n' % (
                        file, que.shape[0], training_accuracy, training_loss))

                # if few == 20 and model == 'HS_CNN':
                #     sup_list_re__ = np.concatenate(
                #         (np.expand_dims(sup_list_re, axis=-1), np.expand_dims(sup_list_re_, axis=-1)), axis=-1)
                #     np.save(save_path.split('result')[0] + '/result/' + key[i].split('.')[0] + '_que_list.npy',
                #             que_list_re)
                #     np.save(save_path.split('result')[0] + '/result/' + key[i].split('.')[0] + '_sup_list.npy',
                #             sup_list_re__)

        f.close()