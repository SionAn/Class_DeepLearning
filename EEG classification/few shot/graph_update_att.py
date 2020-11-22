import sys
sys.path.append('/home/sion/code')
import data_loader
import Code_total as ct
import model as net
import numpy as np
import tensorflow as tf
import os
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import shutil
import math
import pdb

def add_noise(support, num):
    for i in range(support.shape[0]):
        random = np.random.randint(support.shape[1], size=num)
        random_ = np.random.randint(support.shape[1], size=num)
        for j in range(random.shape[0]):
            # if j == 0:
            #     print(np.random.normal(scale = 0.1,size=support[i, random[j], :, :, :, 0].shape))
            support[i, random[j], :, :, :, 0] = support[i, random[j], :, :, :, 0] + \
                                                np.random.normal(scale = 0.1, size=support[i, random[j], :, :, :, 0].shape)
            support[i, random_[j], :, :, :, 1] = support[i, random[j], :, :, :, 1] + \
                                                np.random.normal(scale = 0.1, size=support[i, random[j], :, :, :, 1].shape)
            mean = np.mean(support[i, random[j], :, :, :, 0])
            std = np.std(support[i, random[j], :, :, :, 0])
            support[i, random[j], :, :, :, 0] = (support[i, random[j], :, :, :, 0] - mean) / std
            max = np.max(support[i, random[j], :, :, :, 0])
            min = np.min(support[i, random[j], :, :, :, 0])
            support[i, random[j], :, :, :, 0] = (support[i, random[j], :, :, :, 0] - min) / (max - min)
            mean = np.mean(support[i, random[j], :, :, :, 1])
            std = np.std(support[i, random[j], :, :, :, 1])
            support[i, random[j], :, :, :, 1] = (support[i, random[j], :, :, :, 1] - mean) / std
            max = np.max(support[i, random[j], :, :, :, 1])
            min = np.min(support[i, random[j], :, :, :, 1])
            support[i, random[j], :, :, :, 1] = (support[i, random[j], :, :, :, 1] - min) / (max - min)

    return support, random, random_

def file_copy(input_path, output_path, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    shutil.copy2(input_path+'/'+file_name, output_path+'/'+file_name)

def file_remove(path):
    file_list = os.listdir(path)
    for i in range(len(file_list)):
        if file_list[i][-5:] == '.meta' or file_list[i][-6:] == '.index' or file_list[i][-6:] == '-00001':
            os.remove(os.path.join(path, file_list[i]))

def support_query(batch, few, data_dict):
    data = data_dict['data']
    label = data_dict['label']
    support = np.zeros((batch, few, data.shape[1], data.shape[2], data.shape[3], 2))
    query = np.zeros((batch, 1, data.shape[1], data.shape[2], data.shape[3]))
    lset = np.zeros((batch, label.shape[1]))

    for i in range(batch):
        sup_list = []
        sup_list_ = []
        for j in range(few):
            while True:
                sup = random.randint(0, data.shape[0]-1)
                if np.argmax(label[sup]) == 0:
                    support[i, j, :, :, :, 0] = data[sup]
                    sup_list.append(sup)
                    break
            while True:
                sup_ = random.randint(0, data.shape[0]-1)
                if np.argmax(label[sup_]) == 1:
                    support[i, j, :, :, :, 1] = data[sup_]
                    sup_list_.append(sup_)
                    break
        # while True:
        que = random.randint(0, data.shape[0] - 1)
            # if que not in sup_list and que not in sup_list_:
            #     break
            # else:
            #     print("중복")
            #     print(data.shape[0]-1)
            #     uni, count = np.unique(np.argmax(label, axis = 1), return_counts = 'True')
            #     print(uni, count)

        query[i, 0] = data[que]
        lset[i] = label[que]

    return support, query, lset

def cross_entropy_gat(output, node, y, y_, lrate):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-10)+(1-y)*tf.log(1-output+1e-10), axis=1))  ## cross entropy
    loss += tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(node+1e-10)+(1-y_)*tf.log(1-node+1e-10), axis=1))
    # reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  ## regularizer
    # loss += tf.reduce_sum(reg)
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

def cross_entropy(output, y, lrate):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-6)+(1-y)*tf.log(1-output+1e-6), axis=1))  ## cross entropy
    # loss += tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(node+1e-10)+(1-y_)*tf.log(1-node+1e-10), axis=1))
    # reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  ## regularizer
    # loss += tf.reduce_sum(reg)
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

def Build_graph(train_dict, val_dict, epochs, mini_batch, lr, subject_num, kp, cl, kl, sl, dl, ml, model, few, type):
    with tf.variable_scope('Placeholder'):
        if model == "HS_CNN":
            x = tf.placeholder(tf.float32, shape = [None, few, 875, 3, 3, 2])
            x_ = tf.placeholder(tf.float32, shape = [None, 1, 875, 3, 3])
        else:
            x = tf.placeholder(tf.float32, shape=[None, few, 875, 3, 1, 2])
            x_ = tf.placeholder(tf.float32, shape=[None, 1, 875, 3, 1])

        y = tf.placeholder(tf.float32, shape = [None, 2])
        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.float32, shape = [None, 2])
        lrate = tf.placeholder(tf.float64)
        tr_acc = tf.placeholder(tf.float32)
        tr_loss = tf.placeholder(tf.float32)
        lr_init = lr
        vl_acc = tf.placeholder(tf.float32)
        vl_loss = tf.placeholder(tf.float32)
        prev_loss = 10000000000000
    with tf.variable_scope('Model'):
        # output = RelationNet(x, x_, model, kp)
        # feature_dict = embedding_module(x, x_, model, kp)
        load_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/supervised/' + model + '/subject_' + str(subject_num) + '/' +model
        f = open(load_path + '/log.txt')
        lines = f.readlines()
        file_num = lines[-2].split(':')[1]
        file_num = file_num.split(',')[0]
        file_num = file_num.split(' ')[1]
        file_num = int(file_num)-1
        f.close()
        restore_saver = tf.train.import_meta_graph(load_path+'/model-'+str(file_num)+'.meta')

    with tf.variable_scope('New_Model'):
        if type == '_gcn' or type == '_gat':
            output, Att, Att_, Att__ = net.RelationNet(x, x_, model, kp, type)
        if type == '_att' or type == '_dis':
            output, Att, Att_ = net.RelationNet(x, x_, model, kp, type)
        if type == '_satt':
            output = net.RelationNet(x, x_, model, kp, type)

    with tf.variable_scope('New_Loss'):
        loss, train_step = cross_entropy(output, y, lrate)

    with tf.variable_scope('New_Accuracy'):
        prediction, answer, correct, accuracy = ct.class_accuracy(output, y)

    with tf.variable_scope('New_Tensorboard'):
        with tf.variable_scope('Train'):
            train_summary = [tf.summary.scalar("loss_train", tr_loss), tf.summary.scalar("accuracy_train", tr_acc)]
            summary = tf.summary.merge(train_summary)
        with tf.variable_scope('New_Validation'):
            validation_summary = [tf.summary.scalar("loss_val", vl_loss), tf.summary.scalar("accuracy_val", vl_acc)]
            summary_val = tf.summary.merge(validation_summary)

    ## Save path
    with tf.variable_scope('New_Save_model'):
        now = time.localtime()
        play = "%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        writer = tf.summary.FileWriter(
            '/home/sion/tensorboard/EEG_MI_CLASSIFICATION/BCI4_2b/Fewshot/'+str(few)+'/'+model+type+'/subject_' + str(subject_num) + '/' +model)
        writer.add_graph(tf.get_default_graph())
        save_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/Fewshot/'+str(few)+'/'+model+type+'/subject_' + str(subject_num)+ '/' +model
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        input_path = '/home/sion/code/EEG/meta/BCI4_2b_RN_pretrain'
        ct.file_copy(input_path, save_path, 'main_update.py')
        ct.file_copy(input_path, save_path, 'data_loader.py')
        ct.file_copy(input_path, save_path, 'graph_update_att.py')
        ct.file_copy(input_path, save_path, 'meta_update.sh')

        f = open(save_path + '/log.txt', 'w')

        f.write("Model information\n")
        f.write("Channel             : ")
        f.write("".join(str(cl)))
        f.write("\n")
        f.write("Kernel size          : ")
        f.write("".join(str(kl)))
        f.write("\n")
        f.write("Stride                : ")
        f.write("".join(str(sl)))
        f.write("\n")
        f.write("Maxpooling         : ")
        f.write("".join(str(ml)))
        f.write("\n")
        f.write("FC layer_dimension: ")
        f.write("".join(str(dl)))
        f.write("\n")
        f.write("Learning rate       : %f\n" % (lr))
        f.write("\n")

    ## For log
    with tf.variable_scope('New_Log'):
        current_max_validation = 0
        current_max_validation_index = 0
        current_min_loss = 10000
        total_max_validation = 0
        total_max_validation_index = 0
        total_min_loss = 10000

    ## Open session for Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    val_key = list(val_dict.keys())
    for i in range(len(val_key)):
        if i == 0:
            val_support, val_query, val_lset = support_query(50, few, val_dict[val_key[i]])
        else:
            val_support_, val_query_, val_lset_ = support_query(50, few, val_dict[val_key[i]])
            val_support = np.concatenate((val_support, val_support_), axis = 0)
            val_query = np.concatenate((val_query, val_query_), axis=0)
            val_lset = np.concatenate((val_lset, val_lset_), axis=0)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        restore_saver.restore(sess, load_path + '/model-' + str(file_num))
        op = []
        pretrained = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Graph_build/Model/')
        new_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Graph_build/New_Model/Embedding')
        # for j in range(len(new_model)):
        #     print(pretrained[j].name.split('Graph_build/Model/Graph_build/Model/')[1] == new_model[j].name.split('Graph_build/New_Model/Embedding/')[1])

        for op_ in range(len(new_model)):
            op.append(new_model[op_].assign(pretrained[op_].value()))

        sess.run(op)

        for i in tqdm(range(epochs), 'Epochs'):
            key = list(train_dict.keys())
            train_data = random.randint(0, len(key)-1)
            support, query, lset = support_query(mini_batch, few, train_dict[key[train_data]])
            # print(key[train_data])

            training_accuracy = 0
            training_loss = 0
            # print(batch[1])
            ## Training
            run = 1
            for j in  range(run):
                # tr, aug, aug_ = add_noise(support, 2)
                tr = support
                tr_ = query
                tr_label = lset

                tr_label_ = np.zeros((tr_label.shape[0], 2))
                tr_label_[:, 0] = tr_label[:, 0]
                tr_label_[:, 1] = tr_label[:, 1]
                sess.run(train_step, feed_dict={x: tr, x_: tr_, y: tr_label, y_: tr_label_, keep_prob: kp, lrate: lr})
                predic_value, train_accuracy, loss_print, train_prediction = \
                    sess.run([output, accuracy, loss, prediction],
                             feed_dict={x: tr, x_: tr_, y: tr_label, y_: tr_label_, keep_prob: 1.0})
                training_accuracy += train_accuracy*tr.shape[0]
                training_loss += loss_print*tr.shape[0]

            training_accuracy /= (run * mini_batch)
            training_loss /= (run * mini_batch)

            if i % 1 == 0:
                # Write training accuracy, loss in tensorboard
                al_train = sess.run(summary, feed_dict={tr_acc: training_accuracy, tr_loss: training_loss})
                writer.add_summary(al_train, global_step=i)

                print("")
                print("Epoch: %d, Training Accuracy: %f, loss: %f" % (i + 1, training_accuracy, training_loss))
                print("Label:               ", np.argmax(tr_label[-10:], axis = -1))
                print("Training prediction: ", train_prediction[-10:])
                print(predic_value[-10:])

                # print(attention[-1, 0, :])
                # print(attention_[-1, 0, :])
                #
                # print(attention__[0, -1, :, :few+1]) # gat
                # print(attention__[0, -1, :, few+1:])

                # att_key = list(attention.keys()) # gcn
                # att_list = []
                # att_list_ = []
                # for att_index in range(len(att_key)):
                #     att_list.append(attention[att_key[att_index]][-1, 0])
                #     att_list_.append(attention_[att_key[att_index]][-1, 0])
                # # print(aug)
                # print(att_list)
                # # print(aug_)
                # print(att_list_)

                # att_list = [] # dis
                # att_list_ = []
                # for att_index in range(few):
                #     att_list.append(attention[att_index][-1])
                #     att_list_.append(attention_[att_index][-1])
                # print(att_list)
                # print(att_list_)

                ## Validation
                val_accuracy = 0
                loss_val = 0
                run = int(val_support.shape[0]/mini_batch)
                if run == 0:
                    run = 1
                for j in range(run):
                    vl = val_support[j*mini_batch:(j+1)*mini_batch]
                    vl_ = val_query[j*mini_batch:(j+1)*mini_batch]
                    vl_label = val_lset[j*mini_batch:(j+1)*mini_batch]

                    vl_label_ = np.zeros((vl_label.shape[0], 2))
                    vl_label_[:, 0] = vl_label[:, 0]
                    vl_label_[:, 1] = vl_label[:, 1]

                    predic_value, val_accuracy_bt, loss_val_bt, val_prediction = sess.run(
                        [output, accuracy, loss, prediction],
                        feed_dict={x: vl, x_: vl_, y: vl_label, y_: vl_label_, keep_prob: 1.0})
                    val_accuracy += val_accuracy_bt * vl.shape[0]
                    loss_val += loss_val_bt * vl.shape[0]

                val_accuracy /= (run * vl.shape[0])
                loss_val /= (run * vl.shape[0])

                al_val = sess.run(summary_val, feed_dict={vl_acc: val_accuracy, vl_loss: loss_val})
                writer.add_summary(al_val, global_step=i)

                ## For printing
                if loss_val < current_min_loss:
                    current_max_validation = val_accuracy
                    current_max_validation_index = i
                    current_min_loss = loss_val

                if val_accuracy >= current_max_validation and current_min_loss == loss_val:
                    current_max_validation = val_accuracy
                    current_max_validation_index = i
                    current_min_loss = loss_val

                if (i - current_max_validation_index) >= 10:
                    ct.file_copy(save_path, save_path + '/best',
                                 'model-' + str(current_max_validation_index) + '.data-00000-of-00001')
                    ct.file_copy(save_path, save_path + '/best',
                                 'model-' + str(current_max_validation_index) + '.index')
                    ct.file_copy(save_path, save_path + '/best',
                                 'model-' + str(current_max_validation_index) + '.meta')
                    current_max_validation = val_accuracy
                    current_max_validation_index = i
                    current_min_loss = loss_val

                if current_min_loss < total_min_loss:
                    total_max_validation = current_max_validation
                    total_max_validation_index = current_max_validation_index
                    total_min_loss = current_min_loss

                if current_max_validation >= total_max_validation and current_min_loss == total_min_loss:
                    total_max_validation = current_max_validation
                    total_max_validation_index = current_max_validation_index
                    total_min_loss = current_min_loss

                print("=================================================================================================")
                print("Epoch            : %d, Validation  Accuracy: %f, loss: %f" % (i+1, val_accuracy, loss_val))
                print("Current Max Index: %d, Current Max Accuracy: %f, loss: %f" % (current_max_validation_index + 1,
                                                                                     current_max_validation,
                                                                                     current_min_loss))
                print("total Max Index  : %d, total Max Accuracy: %f, loss: %f" % (total_max_validation_index + 1,
                                                                                     total_max_validation,
                                                                                     total_min_loss))
                # print(attention[-1, 0, :]) # gat
                # print(attention_[-1, 0, :])

                # att_key = list(attention.keys()) # gcn
                # att_list = []
                # att_list_ = []
                # for att_index in range(len(att_key)):
                #     att_list.append(attention[att_key[att_index]][-1, 0])
                #     att_list_.append(attention_[att_key[att_index]][-1, 0])
                # print(att_list)
                # print(att_list_)

                # att_list = [] # dis
                # att_list_ = []
                # for att_index in range(few):
                #     att_list.append(attention[att_index][-1])
                #     att_list_.append(attention_[att_index][-1])
                # print(att_list)
                # print(att_list_)
                print("=================================================================================================")
                # print("")

                f.write(
                    "===============================================================================\n")
                f.write(
                    "Epoch               : %d, Validation  Accuracy  : %f, loss: %f\n" % (i + 1, val_accuracy, loss_val))
                f.write("Current Max Index: %d, Current Max Accuracy: %f, loss: %f\n" % (current_max_validation_index + 1,
                                                                                         current_max_validation,
                                                                                         current_min_loss))
                f.write("total Max Index: %d, total Max Accuracy: %f, loss: %f\n" % (total_max_validation_index + 1,
                                                                                         total_max_validation,
                                                                                         total_min_loss))
                f.write(
                    "===============================================================================\n")

                saver.save(sess, save_path + '/model', global_step=i)

                if i == epochs - 1:
                    ct.file_copy(save_path, save_path + '/best',
                                 'model-' + str(i) + '.data-00000-of-00001')
                    ct.file_copy(save_path, save_path + '/best',
                                 'model-' + str(i) + '.index')
                    ct.file_copy(save_path, save_path + '/best',
                                 'model-' + str(i) + '.meta')

                writer.add_summary(al_val, global_step=i)

                if i % 1 == 0:
                    lr = lr_init * (epochs-(i+1))/epochs
                    print('Learning rate is changed to ', lr)
                print("Model: ", model+type, "N-shot: ", few, "subject: ", subject_num)
                print("")
                if loss_val <= 0.01 or lr < 1e-10 or total_max_validation_index==(i-3000):
                    break


        f.write(
            "===============================================================================\n")
        f.write("Total Max Index: %d, loss: %f\n" % (total_max_validation_index + 1,
                                                     total_min_loss))
        f.write(
            "===============================================================================\n")
        f.close()
        if os.path.isfile(os.path.join(save_path, 'model-' + str(total_max_validation_index) + '.meta')):
            file_copy(save_path, save_path + '/best',
                      'model-' + str(total_max_validation_index) + '.data-00000-of-00001')
            file_copy(save_path, save_path + '/best',
                      'model-' + str(total_max_validation_index) + '.index')
            file_copy(save_path, save_path + '/best',
                      'model-' + str(total_max_validation_index) + '.meta')
        file_remove(save_path)
        file_copy(save_path + '/best', save_path,
                  'model-' + str(total_max_validation_index) + '.data-00000-of-00001')
        file_copy(save_path + '/best', save_path,
                  'model-' + str(total_max_validation_index) + '.index')
        file_copy(save_path + '/best', save_path,
                  'model-' + str(total_max_validation_index) + '.meta')
        file_remove(os.path.join(save_path, 'best'))
        file_copy(save_path, save_path + '/best',
                  'model-' + str(total_max_validation_index) + '.data-00000-of-00001')
        file_copy(save_path, save_path + '/best',
                  'model-' + str(total_max_validation_index) + '.index')
        file_copy(save_path, save_path + '/best',
                  'model-' + str(total_max_validation_index) + '.meta')
