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
import shutil

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
        for j in range(few):
            while True:
                sup = random.randint(0, data.shape[0]-1)
                if np.argmax(label[sup]) == 0:
                    support[i, j, :, :, :, 0] = data[sup]
                    break
            while True:
                sup_ = random.randint(0, data.shape[0]-1)
                if np.argmax(label[sup_]) == 1:
                    support[i, j, :, :, :, 1] = data[sup_]
                    break


        que = random.randint(0, data.shape[0] - 1)
        query[i, 0] = data[que]
        lset[i] = label[que]

    return support, query, lset

def build_model_CNN(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name=''):
    x_data = x
    keep_prob = keep_prob
    pool_count = 0
    for i in range(len(channel_list)):
        x_data = CNN_layer(x_data, channel_list[i], kernel_list[i], stride_list[i], reuse_=re, name=name + str(i),
                           padding='SAME')
        if i % 2 == 1 and i != 0:
            with tf.variable_scope('Dropout-' + str(int(i / 2))):
                # x_data = tf.layers.batch_normalization(x_data, name = 'BatchNorm-'+name+str(int(i/2)))
                x_data = tf.layers.dropout(x_data, rate=keep_prob, name='Dropout-' + name + str(int(i / 2)))
        if i in maxpool_list[::, 0]:
            x_data = tf.layers.average_pooling2d(x_data, pool_size=maxpool_list[pool_count, 1:3],
                                                 strides=maxpool_list[pool_count, 3:5], padding='SAME',
                                                 name='avg-' + name + str(int(i / 2)))
            pool_count += 1

    x_data = tf.layers.flatten(x_data)
    for i in range(len(dense_list)-1):
        x_data = tf.layers.dense(x_data, units = dense_list[i], activation = tf.nn.elu, reuse = re, name = 'FC_layer-'+name+str(int(i)))

    x_data = tf.layers.dense(x_data, units = dense_list[-1], activation = tf.nn.sigmoid, reuse = re, name = 'Sigmoid-'+name)

    return x_data

def RelationNet(support, query, model, kp):
    with tf.variable_scope('Embedding'):
        few = support.shape[1]
        re = False
        for i in range(few):
            if model == 'EEGNet':
                sup_output = build_model_EEGNet(support[:, i, :, :, :, 0], kp, re)
                re = True
                sup_output_ = build_model_EEGNet(support[:, i, :, :, :, 1], kp, re)
            if model == 'ShallowNet':
                sup_output = build_model_ShallowNet(support[:, i, :, :, :, 0], kp, re)
                re = True
                sup_output_ = build_model_ShallowNet(support[:, i, :, :, :, 1], kp, re)
            if model == 'DeepconvNet':
                sup_output = build_model_DeepconvNet(support[:, i, :, :, :, 0], kp, re)
                re = True
                sup_output_ = build_model_DeepconvNet(support[:, i, :, :, :, 1], kp, re)
            if model == 'HS_CNN':
                sup_output = build_model_HS_CNN(support[:, i, :, :, :, 0], kp, re)
                re = True
                sup_output_ = build_model_HS_CNN(support[:, i, :, :, :, 1], kp, re)
            if i == 0:
                output = sup_output/float(int(few))
                output_ = sup_output_/float(int(few))
            if i != 0:
                output = tf.add(output, sup_output/float(int(few)))
                output_ = tf.add(output_, sup_output_/float(int(few)))
        if model == 'EEGNet':
            que_output = build_model_EEGNet(query[:, 0], kp, re)
        if model == 'ShallowNet':
            que_output = build_model_ShallowNet(query[:, 0], kp, re)
        if model == 'DeepconvNet':
            que_output = build_model_DeepconvNet(query[:, 0], kp, re)
        if model == 'HS_CNN':
            que_output = build_model_HS_CNN(query[:, 0], kp, re)

        # output = tf.div(output, tf.constant(few, dtype=tf.float32))
        # output_ = tf.div(output_, tf.constant(few, dtype=tf.float32))

        feature = tf.concat((output, que_output), axis = -1)
        feature_ = tf.concat([output_, que_output], axis = -1)

    with tf.variable_scope('Relation'):
        layer = ct.build_model_CNN(feature, kp, [128, 256], np.asarray([[30, 1], [15, 1]]), [[1, 1], [1, 1]], [512, 256, 64, 1],
                                   np.asarray([[0, 3, 1, 3, 1]]), False, 'Relation')
        layer_ = ct.build_model_CNN(feature_, kp, [128, 256], np.asarray([[30, 1], [15, 1]]), [[1, 1], [1, 1]],
                                   [512, 256, 64, 1],
                                   np.asarray([[0, 3, 1, 3, 1]]), True, 'Relation')

        score = tf.concat((layer, layer_), axis = -1, name = 'output')

    return score

def cross_entropy(output, y, lrate):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-10)+(1-y)*tf.log(1-output+1e-10), axis=1))  ## cross entropy
    # reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  ## regularizer
    # loss += tf.reduce_sum(reg)
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

def CNN_layer_linear(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = None, \
                                  reuse=reuse_, kernel_initializer=tf.random_normal_initializer(stddev=0.0001, mean=0))
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.contrib.layers.xavier_initializer(),

        return output

def CNN_layer_relation(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = tf.nn.elu, \
                                  reuse=reuse_, kernel_initializer=tf.contrib.layers.xavier_initializer())
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.random_normal_initializer(stddev=0.0001, mean=0),

        return output

def build_model_DeepconvNet(input, kp, reuse):
    with tf.variable_scope("Block1"):
        layer = CNN_layer_linear(input, 25, [10, 1], [1, 1], reuse, 'conv1', 'valid')
        layer = CNN_layer_linear(layer, 25, [1, 3], [1, 1], reuse, 'Depthconv', 'valid')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm1')
        layer = tf.nn.elu(layer)
        # layer = tf.layers.max_pooling2d(layer, pool_size = [3, 1], strides = [3, 1], padding = 'valid',
        #                                          name = 'maxpool1')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout1')
    with tf.variable_scope("Block2"):
        layer = CNN_layer_linear(layer, 50, [10, 1], [1, 1], reuse, 'conv2', 'valid')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm2')
        layer = tf.nn.elu(layer)
        layer = tf.layers.max_pooling2d(layer, pool_size=[3, 1], strides=[3, 1], padding='valid',
                                        name='maxpool2')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout2')
    with tf.variable_scope("Block3"):
        layer = CNN_layer_linear(layer, 100, [10, 1], [1, 1], reuse, 'conv3', 'valid')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm3')
        layer = tf.nn.elu(layer)
        # layer = tf.layers.max_pooling2d(layer, pool_size=[3, 1], strides=[3, 1], padding='valid',
        #                                 name='maxpool3')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout3')
    with tf.variable_scope("Block4"):
        layer = CNN_layer_linear(layer, 200, [10, 1], [1, 1], reuse, 'conv4', 'valid')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm4')
        layer = tf.nn.elu(layer)
        layer = tf.layers.max_pooling2d(layer, pool_size=[3, 1], strides=[3, 1], padding='valid',
                                        name='maxpool4')
    #     layer = tf.layers.dropout(layer, rate=kp, name='dropout4')
    # layer = tf.layers.flatten(layer)
    # layer = tf.layers.dense(layer, units=2, activation=tf.nn.softmax, name='Softmax')

    return layer

def build_model_ShallowNet(input, kp, reuse):
    layer = CNN_layer_linear(input, 40, [25, 1], [1, 1], reuse, 'conv', 'same')
    layer = CNN_layer_linear(layer, 40, [1, 3], [1, 1], reuse, 'Depthconv', 'valid')
    layer = tf.layers.batch_normalization(layer, reuse = reuse, name = 'BatchNorm')
    layer = tf.square(layer)
    layer = tf.layers.average_pooling2d(layer, pool_size = [75, 1], strides = [7, 1], padding = 'valid',
                                             name = 'avgpool')
    layer = tf.log(layer)
    # layer = tf.layers.flatten(layer)
    # layer = tf.layers.dropout(layer, rate = kp, name = 'dropout')
    # layer = tf.layers.dense(layer, units=2, activation=tf.nn.softmax, name='Softmax')

    return layer

def build_model_EEGNet(input, kp, reuse):
    with tf.variable_scope('Block1'):
        layer = CNN_layer_linear(input, 8, [125, 1], [1, 1], reuse, 'conv', 'same')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm')
        layer = CNN_layer_linear(layer, 16, [1, 3], [1, 1], reuse, 'Depthconv', 'valid')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm-Depth')
        layer = tf.nn.elu(layer)
        layer = tf.layers.average_pooling2d(layer, pool_size=[4, 1], strides=[2, 1], padding='same',
                                            name='avgpool')
        layer = tf.layers.dropout(layer, rate=kp, name='dropout')

    with tf.variable_scope('Block2'):
        layer = CNN_layer_linear(layer, 16, [16, 1], [1, 1], reuse, 'conv', 'same')
        layer = tf.layers.batch_normalization(layer, reuse=reuse, name='BatchNorm')
        layer = tf.nn.elu(layer)
        layer = tf.layers.average_pooling2d(layer, pool_size=[8, 1], strides=[4, 1], padding='same',
                                            name='avgpool')
    #     layer = tf.layers.dropout(layer, rate=kp, name='dropout')
    #
    # layer = tf.layers.flatten(layer)
    # layer = tf.layers.dense(layer, units=2, activation=tf.nn.softmax, name='Softmax')

    return layer

def CNN_layer(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = tf.nn.elu, \
                                  reuse=reuse_, kernel_initializer=tf.random_normal_initializer(stddev=0.0001, mean=0))
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.contrib.layers.xavier_initializer(),

        return output

def build_model_HS_CNN(input, kp, reuse, name = ''):
    # input = input * 10000000

    for i in range(3):
        input_ = input[::, ::, ::, i:i + 1]
    # j = 0
    # input1 = input[::, ::, ::, 0:1]
    # input2 = input[::, ::, ::, 1:2]
    # input3 = input[::, ::, ::, 2:3]
        with tf.variable_scope('Freq'+str(i)):
            layer1 = CNN_layer(input_, 10, [45, 1], [3, 1], reuse_ = reuse, name = name+'45_'+str(i), padding='SAME')
            layer2 = CNN_layer(input_, 10, [65, 1], [3, 1], reuse_ = reuse, name = name+'65_'+str(i), padding='SAME')
            layer3 = CNN_layer(input_, 10, [85, 1], [3, 1], reuse_ = reuse, name = name+'85_'+str(i), padding='SAME')

            layer1 = CNN_layer(layer1, 10, [1, 3], [1, 1], reuse_=reuse, name=name + 'c45_' +str(i))
            layer2 = CNN_layer(layer2, 10, [1, 3], [1, 1], reuse_=reuse, name=name + 'c65_' +str(i))
            layer3 = CNN_layer(layer3, 10, [1, 3], [1, 1], reuse_=reuse, name=name + 'c85_' +str(i))

            layer1 = tf.layers.max_pooling2d(layer1, pool_size = [6, 1], strides = [6, 1], padding = 'VALID',
                                             name = 'max1-'+name+str(i))
            layer2 = tf.layers.max_pooling2d(layer2, pool_size=[6, 1], strides=[6, 1], padding='VALID',
                                             name='max2-' + name +str(i))
            layer3 = tf.layers.max_pooling2d(layer3, pool_size=[6, 1], strides=[6, 1], padding='VALID',
                                             name='max3-' + name +str(i))

    #         layer1 = tf.layers.flatten(layer1)
    #         layer2 = tf.layers.flatten(layer2)
    #
    #         layer3 = tf.layers.flatten(layer3)
    #
            if i == 0:
                layer = tf.concat([layer1, layer2, layer3], axis = -1)

            else:
                layer = tf.concat([layer, layer1, layer2, layer3], axis = -1)
    #
    # for i in range(len(dl) - 1):
    #     layer = tf.layers.dense(layer, units=dl[i], activation=tf.nn.elu,
    #                             name='FC_layer-' + name + str(int(i)))
    #     layer = tf.layers.dropout(layer, rate=kp, name='Dropout-' + name + str(int(i / 2)))
    #
    # layer = tf.layers.dense(layer, units=dl[-1], activation=tf.nn.softmax, name='Softmax-' + name)

    return layer

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
        load_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/supervised/' + model + '/subject_' + str(subject_num) + '/'+model
        f = open(load_path + '/log.txt')
        lines = f.readlines()
        file_num = lines[-2].split(':')[1]
        file_num = file_num.split(',')[0]
        file_num = file_num.split(' ')[1]
        file_num = int(file_num)-1
        f.close()
        restore_saver = tf.train.import_meta_graph(load_path+'/model-'+str(file_num)+'.meta')

    with tf.variable_scope('New_Model'):
        output = RelationNet(x, x_, model, kp)

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
            '/home/sion/tensorboard/EEG_MI_CLASSIFICATION/BCI4_2b/Fewshot/'+str(few)+'/'+model+type+'/subject_' + str(subject_num) + '/HS')
        writer.add_graph(tf.get_default_graph())
        save_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/Fewshot/'+str(few)+'/'+model+type+'/subject_' + str(subject_num)+ '/HS'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        input_path = '/home/sion/code/EEG/meta/BCI4_2b_RN_pretrain'
        ct.file_copy(input_path, save_path, 'main_update.py')
        ct.file_copy(input_path, save_path, 'data_loader.py')
        ct.file_copy(input_path, save_path, 'graph_update.py')
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
            val_support, val_query, val_lset = support_query(100, few, val_dict[val_key[i]])
        else:
            val_support_, val_query_, val_lset_ = support_query(100, few, val_dict[val_key[i]])
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
                tr = support
                tr_ = query
                tr_label = lset

                sess.run(train_step, feed_dict={x: tr, x_: tr_, y: tr_label, keep_prob: kp, lrate: lr})
                predic_value, train_accuracy, loss_print, train_prediction = \
                    sess.run([output, accuracy, loss, prediction], feed_dict= {x: tr, x_: tr_, y: tr_label, keep_prob: 1.0})
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

                    predic_value, val_accuracy_bt, loss_val_bt, val_prediction = sess.run(
                        [output, accuracy, loss, prediction], feed_dict= {x: vl, x_: vl_, y: vl_label, keep_prob: 1.0})
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
                print("Model: ", model, "N-shot: ", few, "subject: ", subject_num)
                print("")
                if loss_val <= 0.01 or lr < 1e-10 or val_accuracy >= 0.9 or total_max_validation_index==(i-3000):
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
        os.rmdir(os.path.join(save_path, 'best'))
