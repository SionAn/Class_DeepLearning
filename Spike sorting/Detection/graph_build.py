import model
import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm
import shutil
import random

def normalization(data): # [N, W, H, C] signal = N, 32, 1, 2
    for i in range(data.shape[0]):
        for j in range(data.shape[-1]):
            minimum = np.min(data[i, :, :, j])
            maximum = np.max(data[i, :, :, j])
            data[i, :, :, j] = (data[i, :, :, j]-minimum)/(maximum-minimum)

    return data

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

def class_accuracy(output, y):
    prediction = tf.argmax(output, 1)
    answer = tf.argmax(y, 1)
    correct = tf.equal(prediction, answer)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return prediction, answer, correct, accuracy

def next_batch(num, data1, labels1):
    with tf.variable_scope('Suffle'):
        idx = np.arange(0, data1.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle1 = [data1[i] for i in idx]

        labels_shuffle1 = [labels1[i] for i in idx]

    return np.asarray(data_shuffle1), np.asarray(labels_shuffle1)

def Build_graph(train_dict, val_dict, kernel, channel, stride, dense, pooling, lr, kp, epochs, mini_batch, save_path, code_path):
    with tf.variable_scope('Placeholder'):
        x = tf.placeholder(tf.float32, shape = [None, 32, 1, 2])
        y = tf.placeholder(tf.float32, shape = [None, 2])
        keep_prob = tf.placeholder(tf.float32)
        tr_acc = tf.placeholder(tf.float32)
        tr_loss = tf.placeholder(tf.float32)
        lrate = tf.placeholder(tf.float32)
        vl_acc = tf.placeholder(tf.float32)
        vl_loss = tf.placeholder(tf.float32)
        lr_init = lr

    with tf.variable_scope('Model'):
        output = model.build_model_CNN(x, keep_prob, channel, kernel, stride, dense, pooling, re = False, name = '')

    with tf.variable_scope('Loss'):
        loss, train_step = cross_entropy(output, y, lrate)

    with tf.variable_scope('Accuracy'):
        prediction, answer, correct, accuracy = class_accuracy(output, y)

    with tf.variable_scope('Tensorboard'):
        with tf.variable_scope('Train'):
            train_summary = [tf.summary.scalar("loss_train", tr_loss), tf.summary.scalar("accuracy_train", tr_acc)]
            summary = tf.summary.merge(train_summary)
        with tf.variable_scope('Validation'):
            validation_summary = [tf.summary.scalar("loss_val", vl_loss), tf.summary.scalar("accuracy_val", vl_acc)]
            summary_val = tf.summary.merge(validation_summary)

    with tf.variable_scope('Save_model'):
        now = time.localtime()
        play = "%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        writer = tf.summary.FileWriter(os.path.join(save_path, 'tensorboard/Spike_sorting/detection/'+play))
        writer.add_graph(tf.get_default_graph())
        save_path = os.path.join(save_path, 'checkpoint/Spike_sorting/detection/'+play)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_copy(code_path, save_path, 'main.py')
        file_copy(code_path, save_path, 'data_loader.py')
        file_copy(code_path, save_path, 'graph_build.py')
        file_copy(code_path, save_path, 'model.py')

        f = open(save_path + '/log.txt', 'w')

        f.write("Model information\n")
        f.write("Channel             : ")
        f.write("".join(str(channel)))
        f.write("\n")
        f.write("Kernel size          : ")
        f.write("".join(str(kernel)))
        f.write("\n")
        f.write("Stride                : ")
        f.write("".join(str(stride)))
        f.write("\n")
        f.write("pooling                : ")
        f.write("".join(str(np.ndarray.tolist(pooling))))
        f.write("\n")
        f.write("FC layer_dimension: ")
        f.write("".join(str(dense)))
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
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
            train_key = list(train_dict.keys())
            val_key = list(val_dict.keys())
            data_val = []
            label_val = []
            for i in range(len(val_key)):
                data_val.append(np.concatenate(val_dict[val_key[i]]['signal'], axis = 0))
                label_c = np.concatenate(val_dict[val_key[i]]['label'], axis = 0)
                label = np.zeros((label_c.shape[0], 2))
                for j in range(label_c.shape[0]):
                    if label_c[j, 0] == 1.0:
                        label[j, 0] = 1
                    else:
                        label[j, 1] = 1
                label_val.append(label)
            data_val = normalization(np.concatenate(data_val, axis = 0))
            label_val = np.concatenate(label_val, axis = 0)

            for i in tqdm(range(epochs), 'Epochs'):
                training_accuracy = 0
                training_loss = 0
                training_count = 0
                for item in range(len(train_key)):
                    data_num = np.zeros(2, dtype = int)
                    signal = np.concatenate(train_dict[train_key[item]]['signal'][1:], axis = 0)
                    noise = train_dict[train_key[item]]['signal'][0]
                    if signal.shape[0] >= noise.shape[0]:
                        train_list = random.sample(list(range(signal.shape[0])), noise.shape[0])
                        signal = signal[train_list]

                    else:
                        train_list = random.sample(list(range(noise.shape[0])), signal.shape[0])
                        noise = noise[train_list]

                    label_n = np.zeros((noise.shape[0], 2))
                    label_n[:, 0] = 1
                    label_s = np.zeros((signal.shape[0], 2))
                    label_s[:, 1] = 1
                    data = np.concatenate((signal, noise), axis = 0)
                    label = np.concatenate((label_s, label_n), axis = 0)
                    batch = next_batch(data.shape[0], data, label)

                    # print(batch[1])
                    ## Training
                    run = int(data.shape[0] / mini_batch)
                    if data.shape[0] % mini_batch != 0:
                        run += 1
                    for j in range(run):
                        tr = normalization(batch[0][j * mini_batch:(j + 1) * mini_batch])
                        tr_label = batch[1][j * mini_batch:(j + 1) * mini_batch]

                        sess.run(train_step, feed_dict={x: tr, y: tr_label, keep_prob: kp, lrate: lr})
                        predic_value, train_accuracy, loss_print, train_prediction = \
                            sess.run([output, accuracy, loss, prediction],
                                     feed_dict={x: tr, y: tr_label, keep_prob: 1.0})
                        training_accuracy += train_accuracy * tr.shape[0]
                        training_loss += loss_print * tr.shape[0]
                        training_count += tr.shape[0]

                training_accuracy /= training_count
                training_loss /= training_count

                if i % 10 == 9:
                    ## Write training accuracy, loss in tensorboard
                    al_train = sess.run(summary, feed_dict={tr_acc: training_accuracy, tr_loss: training_loss})
                    writer.add_summary(al_train, global_step=i)

                    print("")
                    print("Epoch: %d, Training Accuracy: %f, loss: %f" % (i + 1, training_accuracy, training_loss))
                    print("Label:               ", np.argmax(tr_label[-10:], axis=-1))
                    print("Training prediction: ", train_prediction[-10:])
                    print(predic_value[-10:])

                    ## Validation
                    val_accuracy = 0
                    loss_val = 0
                    run = int(data_val.shape[0] / mini_batch)
                    if data_val.shape[0] % mini_batch != 0:
                        run += 1
                    for j in range(run):
                        vl = data_val[j * mini_batch:(j + 1) * mini_batch]
                        vl_label = label_val[j * mini_batch:(j + 1) * mini_batch]

                        predic_value, val_accuracy_bt, loss_val_bt, val_prediction = sess.run(
                            [output, accuracy, loss, prediction], feed_dict={x: vl, y: vl_label, keep_prob: 1.0})
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

                    if (i - current_max_validation_index) >= 100:
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(current_max_validation_index) + '.data-00000-of-00001')
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(current_max_validation_index) + '.index')
                        file_copy(save_path, save_path + '/best',
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

                    print(
                        "=================================================================================================")
                    print("Epoch            : %d, Validation  Accuracy: %f, loss: %f" % (i + 1, val_accuracy, loss_val))
                    print(
                        "Current Max Index: %d, Current Max Accuracy: %f, loss: %f" % (current_max_validation_index + 1,
                                                                                       current_max_validation,
                                                                                       current_min_loss))
                    print("total Max Index  : %d, total Max Accuracy: %f, loss: %f" % (total_max_validation_index + 1,
                                                                                       total_max_validation,
                                                                                       total_min_loss))
                    print(
                        "=================================================================================================")
                    # print("")

                    f.write(
                        "===============================================================================\n")
                    f.write(
                        "Epoch               : %d, Validation  Accuracy  : %f, loss: %f\n" % (
                        i + 1, val_accuracy, loss_val))
                    f.write("Current Max Index: %d, Current Max Accuracy: %f, loss: %f\n" % (
                    current_max_validation_index + 1,
                    current_max_validation,
                    current_min_loss))
                    f.write("total Max Index: %d, total Max Accuracy: %f, loss: %f\n" % (total_max_validation_index + 1,
                                                                                         total_max_validation,
                                                                                         total_min_loss))
                    f.write(
                        "===============================================================================\n")

                    saver.save(sess, save_path + '/model', global_step=i)

                    if i == epochs - 1:
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(i) + '.data-00000-of-00001')
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(i) + '.index')
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(i) + '.meta')

                    writer.add_summary(al_val, global_step=i)

                    lr = lr_init * (epochs-(i+1))/epochs
                    print('Learning rate is changed to ', lr)

                    print("")
                    # if loss_val > 20:
                    #     break
                if i-total_max_validation_index >= 3000:
                    break
                if training_loss <= 0.01 or training_accuracy >= 0.99:
                    break

            file_copy(save_path, save_path + '/best',
                         'model-' + str(current_max_validation_index) + '.data-00000-of-00001')
            file_copy(save_path, save_path + '/best',
                         'model-' + str(current_max_validation_index) + '.index')
            file_copy(save_path, save_path + '/best',
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