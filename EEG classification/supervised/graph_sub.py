import sys
sys.path.append('/home/sion/code')
import data_loader
import Code_total as ct
import numpy as np
import tensorflow as tf
import os
import time
from tqdm import tqdm
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

def cross_entropy(output, y, lrate):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-10)+(1-y)*tf.log(1-output+1e-10), axis=1))  ## cross entropy
    reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  ## regularizer
    loss += tf.reduce_sum(reg)
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

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

def CNN_layer(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = tf.nn.elu, \
                                  reuse=reuse_, kernel_initializer=tf.random_normal_initializer(stddev=0.0001, mean=0))
                                  # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.contrib.layers.xavier_initializer(),

        return output

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

def Build_graph(train_dict, val_dict, epochs, mini_batch, lr, subject_num, kp, cl, kl, sl, dl, ml, model):
    with tf.variable_scope('Placeholder'):
        if model == "HS_CNN" or model == "HS_CNN_IROS":
            x = tf.placeholder(tf.float32, shape = [None, 875, 3, 3])
        else:
            x = tf.placeholder(tf.float32, shape=[None, 875, 3, 1])
        y = tf.placeholder(tf.float32, shape = [None, 2])
        keep_prob = tf.placeholder(tf.float32)
        tr_acc = tf.placeholder(tf.float32)
        tr_loss = tf.placeholder(tf.float32)
        lr_init = lr
        lrate = tf.placeholder(tf.float32)
        vl_acc = tf.placeholder(tf.float32)
        vl_loss = tf.placeholder(tf.float32)
        prev_loss = 10000000000000

    with tf.variable_scope('Model'):
        if model == 'HS_CNN' or model == 'HS_CNN_IROS':
            output = build_model_HS_CNN(x, keep_prob, cl, sl, dl, re=False, name='')
        if model == 'EEGNet':
            output = build_model_EEGNet(x, keep_prob)
        if model == 'ShallowNet':
            output = build_model_ShallowNet(x, keep_prob)
        if model == 'DeepconvNet':
            output = build_model_DeepconvNet(x, keep_prob)

    with tf.variable_scope('Loss'):
        # lrate = tf.train.exponential_decay(lr, 1, epochs, (1 - 1 / epochs), True)
        loss, train_step = cross_entropy(output, y, lrate)

    with tf.variable_scope('Accuracy'):
        prediction, answer, correct, accuracy = ct.class_accuracy(output, y)

    with tf.variable_scope('Tensorboard'):
        with tf.variable_scope('Train'):
            train_summary = [tf.summary.scalar("loss_train", tr_loss), tf.summary.scalar("accuracy_train", tr_acc)]
            summary = tf.summary.merge(train_summary)
        with tf.variable_scope('Validation'):
            validation_summary = [tf.summary.scalar("loss_val", vl_loss), tf.summary.scalar("accuracy_val", vl_acc)]
            summary_val = tf.summary.merge(validation_summary)

    ## Save path
    with tf.variable_scope('Save_model'):
        now = time.localtime()
        play = "%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        writer = tf.summary.FileWriter(
            '/home/sion/tensorboard/EEG_MI_CLASSIFICATION/BCI4_2b/supervised/'+model+'/subject_' + str(subject_num) + '/'+model)
        writer.add_graph(tf.get_default_graph())
        save_path = '/home/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/supervised/'+model+'/subject_' + str(subject_num)+ '/'+model
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        input_path = '/home/sion/code/EEG/meta/BCI4_2b'
        ct.file_copy(input_path, save_path, 'main_sub.py')
        ct.file_copy(input_path, save_path, 'data_loader.py')
        ct.file_copy(input_path, save_path, 'graph_sub.py')
        ct.file_copy(input_path, save_path, 'meta.sh')
        # np.save(save_path+'/Riem.npy', Riem)
        # np.save(save_path+'/data_test.npy', data_test)
        # np.save(save_path+'/label_test.npy', label_test)

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
    with tf.variable_scope('Log'):
        current_max_validation = 0
        current_max_validation_index = 0
        current_min_loss = 10000
        total_max_validation = 0
        total_max_validation_index = 0
        total_min_loss = 10000

    ## Open session for Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)
        train_key = list(train_dict.keys())
        val_key = list(val_dict.keys())
        for i in range(len(val_key)):
            if i == 0:
                data_val = val_dict[val_key[i]]['data']
                label_val = val_dict[val_key[i]]['label']
            else:
                data_val = np.concatenate((data_val, val_dict[val_key[i]]['data']), axis = 0)
                label_val = np.concatenate((label_val, val_dict[val_key[i]]['label']), axis = 0)

        for i in tqdm(range(epochs), 'Epochs'):
            training_accuracy = 0
            training_loss = 0
            training_count = 0
            for item in range(len(train_key)):
                data = train_dict[train_key[item]]['data']
                label = train_dict[train_key[item]]['label']
                batch = ct.next_batch(data.shape[0], data, label)

                # print(batch[1])
                ## Training
                run = int(data.shape[0]/mini_batch)
                if data.shape[0]%mini_batch != 0:
                    run += 1
                for j in  range(run):
                    tr = batch[0][j*mini_batch:(j+1)*mini_batch]
                    tr_label = batch[1][j*mini_batch:(j+1)*mini_batch]

                    sess.run(train_step, feed_dict={x: tr, y: tr_label, keep_prob: kp, lrate:lr})
                    predic_value, train_accuracy, loss_print, train_prediction = \
                        sess.run([output, accuracy, loss, prediction], feed_dict= {x: tr, y: tr_label, keep_prob: 1.0})
                    training_accuracy += train_accuracy*tr.shape[0]
                    training_loss += loss_print*tr.shape[0]
                    training_count += tr.shape[0]

            training_accuracy /= training_count
            training_loss /= training_count

            if i % 1 == 0:
                ## Write training accuracy, loss in tensorboard
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
                run = int(data_val.shape[0]/mini_batch)
                if data_val.shape[0]%mini_batch != 0:
                    run += 1
                for j in range(run):
                    vl = data_val[j*mini_batch:(j+1)*mini_batch]
                    vl_label = label_val[j*mini_batch:(j+1)*mini_batch]

                    predic_value, val_accuracy_bt, loss_val_bt, val_prediction = sess.run(
                        [output, accuracy, loss, prediction], feed_dict= {x: vl, y: vl_label, keep_prob: 1.0})
                    val_accuracy += val_accuracy_bt * vl.shape[0]
                    loss_val += loss_val_bt * vl.shape[0]

                val_accuracy /= data_val.shape[0]
                loss_val /= data_val.shape[0]

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
                print("Model: ", model, " Subejct: ", subject_num)
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

                # if i % 1 == 0:
                #     lr = lr_init * (epochs-(i+1))/epochs
                #     print('Learning rate is changed to ', lr)

                print("")
                if loss_val > 5 or total_max_validation_index==(i-3000):
                    break

            if training_loss <= 0.01 or training_accuracy >= 0.99:
                break

        ct.file_copy(save_path, save_path + '/best',
                     'model-' + str(current_max_validation_index) + '.data-00000-of-00001')
        ct.file_copy(save_path, save_path + '/best',
                     'model-' + str(current_max_validation_index) + '.index')
        ct.file_copy(save_path, save_path + '/best',
                     'model-' + str(current_max_validation_index) + '.meta')

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
