import model
import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm
import shutil
import random
import pdb

def dict_generate(x, batch, dict_num):
    x_idx = list(range(x.shape[0]))
    batch_idx = random.sample(x_idx, batch)
    output = []
    for b in range(batch):
        ex = list(range(x.shape[0]))
        ex.remove(batch_idx[b])
        dict_idx = random.sample(ex, dict_num)
        if batch_idx[b] in dict_idx:
            dict_idx.remove(batch_idx[b])

        dict_idx.insert(0, batch_idx[b])
        sample = x[batch_idx[b]:batch_idx[b]+1]
        sample = np.concatenate((add_noise(sample), add_noise(x[dict_idx])), axis = 0)
        output.append(np.expand_dims(sample, axis = 0))

    return np.concatenate(output, axis =0)

def add_noise(x, scale_range = 0.03):
    scale_ = np.random.uniform(0, scale_range, x.shape[0]*2)
    size_ = x[0, :, 0, 0].shape[0]
    for aug in range(x.shape[0]):
        x_ = x[aug, :, 0, 0] + np.random.normal(scale = scale_[2*aug], size = size_)
        minimum = np.min(x_)
        maximum = np.max(x_)
        x[aug, :, 0, 0] = (x_-minimum)/(maximum-minimum)

    return x

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

def xentropy(query, key, lrate, temp):
    l_pos = tf.expand_dims(tf.reduce_sum(query * key[0], axis=1), axis=-1)
    l_neg = tf.transpose(tf.reduce_sum(query*key[1:], axis = -1), [1, 0])
    logits = tf.concat([l_pos, l_neg], axis = 1)
    logits = logits*(1/temp)
    loss = tf.reduce_mean(-tf.log(tf.exp(l_pos/temp)/tf.expand_dims(tf.reduce_sum(tf.exp(logits), axis = 1), axis = 1)))
    train_step = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)

    return loss, train_step

def UpdateMomentumEncoder(op, momentum=0.999, setup = False):
    if setup == True:
        var_mapping = {}  # var -> mom var
        nontrainable_vars = tf.trainable_variables() #tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))))
        all_vars = {v.name: v for v in tf.global_variables() + tf.local_variables()}
        # find variables of encoder & momentum encoder

        momentum_prefix = "Graph_build/Model/momentum/"
        for mom_var in nontrainable_vars:
            if momentum_prefix in mom_var.name:
                q_encoder_name = mom_var.name.replace(momentum_prefix, "Graph_build/Model/")
                q_encoder_var = all_vars[q_encoder_name]
                assert q_encoder_var not in var_mapping
                if not q_encoder_var.trainable:  # don't need to copy EMA
                    continue
                var_mapping[q_encoder_var] = mom_var

        assign_ops = [tf.assign(mom_var, var) for var, mom_var in var_mapping.items()]
        assign_op = tf.group(assign_ops, name="initialize_momentum_encoder")

        update_ops = [tf.assign_add(mom_var, (var - mom_var) * (1 - momentum))
                      for var, mom_var in var_mapping.items()]
        update_op = tf.group(update_ops, name="update_momentum_encoder")
        assign_op.run()

        return update_op
    else:
        update_op = op
        update_op.run()

        return update_op

def class_accuracy(output, y):
    prediction = tf.argmax(output, 1)
    answer = tf.argmax(y, 1)
    correct = tf.equal(prediction, answer)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return prediction, answer, correct, accuracy

def next_batch(num, data1):
    with tf.variable_scope('Suffle'):
        idx = np.arange(0, data1.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle1 = [data1[i] for i in idx]

    return np.asarray(data_shuffle1)

def Build_graph(train_dict, val_dict, train_idx, val_idx, kernel, channel, stride, dense, pooling, lr, kp, epochs, mini_batch, save_path,
                code_path, dict_num):
    with tf.variable_scope('Placeholder'):
        x = tf.placeholder(tf.float32, shape = [None, 32, 1, 1])
        x_ = tf.placeholder(tf.float32, shape=[None, 32, 1, 1])
        key_list = tf.placeholder(tf.float32, shape = [dict_num+1, None, dense[-1]])
        keep_prob = tf.placeholder(tf.float32)
        tr_loss = tf.placeholder(tf.float32)
        lrate = tf.placeholder(tf.float32)
        vl_loss = tf.placeholder(tf.float32)
        lr_init = lr

    with tf.variable_scope('Model'):
        query = model.build_model_query(x, keep_prob, channel, kernel, stride, dense, pooling, re = False, name = '')
        key = model.build_model_key(x_, keep_prob, channel, kernel, stride, dense, pooling, re = False, name = '')
    with tf.variable_scope('Loss'):
        loss, train_step = xentropy(query, key_list, lrate, 0.07)

    with tf.variable_scope('Tensorboard'):
        with tf.variable_scope('Train'):
            train_summary = [tf.summary.scalar("loss_train", tr_loss)]
            # summary = tf.summary.merge(train_summary)

        with tf.variable_scope('Validation'):
            validation_summary = [tf.summary.scalar("loss_val", vl_loss)]
            # summary_val = tf.summary.merge(validation_summary)

        for train_var in tf.trainable_variables():
            tf.summary.histogram(train_var.name, train_var)
        weight = tf.summary.merge_all()

    with tf.variable_scope('Save_model'):
        now = time.localtime()
        play = "%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        writer = tf.summary.FileWriter(os.path.join(save_path, 'tensorboard/Spike_sorting/representation/'+play))
        writer.add_graph(tf.get_default_graph())
        save_path = os.path.join(save_path, 'checkpoint/Spike_sorting/representation/'+play)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_copy(code_path, save_path, 'main.py')
        file_copy(code_path, save_path, 'data_loader.py')
        file_copy(code_path, save_path, 'graph_build.py')
        file_copy(code_path, save_path, 'model.py')

        f = open(save_path + '/log.txt', 'w')
        f.write("Training information\n")
        f.write("Batch size: ")
        f.write("".join(str(mini_batch)))
        f.write("\n")
        f.write("Dictionary size: ")
        f.write("".join(str(dict_num)))
        f.write("\n\n")
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
            for i in range(len(val_key)):
                data_val_ = val_dict[val_key[i]]['signal']
                idx = val_idx[val_key[i]]
                for j in range(1, len(data_val_)):
                    data_val.append(data_val_[j][idx[j]])
            data_val = np.concatenate(data_val, axis = 0)
            data_val = data_val[:, :, :, 0:1]

            for i in range(epochs):
                print('Epoch:', int(i+1), '/', epochs)
                training_loss = 0
                training_count = 0
                for item in tqdm(range(len(train_key)), 'File'):
                    signal = []
                    signal_ = train_dict[train_key[item]]['signal']
                    idx = train_idx[train_key[item]]
                    for cluster in range(len(signal_)):
                        signal.append(signal_[cluster][idx[cluster]])
                    data = np.concatenate(signal, axis = 0)
                    data = data[:, :, :, 0:1]
                    data = dict_generate(data, mini_batch, dict_num)

                    # print(batch[1])
                    ## Training
                    run = int(data.shape[0] / mini_batch)
                    if data.shape[0] % mini_batch != 0:
                        run += 1
                    for j in range(run):
                        tr = data[j * mini_batch:(j + 1) * mini_batch]
                        keys = []
                        for k in range(dict_num+1):
                            tr_key = sess.run(key, feed_dict={x_: tr[:, 1+k], keep_prob: kp})
                            keys.append(tr_key)
                        sess.run(train_step, feed_dict={x: tr[:, 0], key_list: keys, keep_prob: kp, lrate: lr})
                        if i == 0 and item == 0:
                            update = 0
                            setup = True
                        update = UpdateMomentumEncoder(update, 0.999, setup)
                        setup = False
                        loss_print = \
                            sess.run(loss,
                                     feed_dict={x: tr[:, 0], key_list: keys, keep_prob: 1.0})
                        training_loss += loss_print * tr.shape[0]
                        training_count += tr.shape[0]

                training_loss /= training_count

                if i % 1 == 0:
                    ## Write training accuracy, loss in tensorboard
                    # al_train = sess.run(summary, feed_dict={tr_loss: training_loss})
                    # writer.add_summary(al_train, global_step=i)

                    print("")
                    print("Epoch: %d, loss: %f" % (i + 1, training_loss))

                    ## Validation

                    loss_val = 0
                    run = int(data_val.shape[0] / mini_batch)
                    if data_val.shape[0] % mini_batch != 0:
                        run += 1
                    for j in range(run):
                        vl = data_val[j * mini_batch:(j + 1) * mini_batch]
                        vl_key = sess.run(key, feed_dict={x_: add_noise(vl), keep_prob: 1.0})
                        keys[0] = vl_key
                        if j == run-1:
                            for k in range(len(keys)):
                                keys[k] = keys[k][:vl.shape[0]]
                        loss_val_bt = sess.run(loss, feed_dict={x: add_noise(vl), key_list: keys, keep_prob: 1.0})
                        loss_val += loss_val_bt * vl.shape[0]

                    loss_val /= data_val.shape[0]

                    # al_val = sess.run(summary_val, feed_dict={vl_loss: loss_val})
                    # writer.add_summary(al_val, global_step=i)
                    weight_train = sess.run(weight, feed_dict={vl_loss: loss_val, tr_loss: training_loss})
                    writer.add_summary(weight_train, global_step=i)

                    ## For printing
                    if loss_val < current_min_loss:
                        current_max_validation_index = i
                        current_min_loss = loss_val

                    if (i - current_max_validation_index) >= 10:
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(current_max_validation_index) + '.data-00000-of-00001')
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(current_max_validation_index) + '.index')
                        file_copy(save_path, save_path + '/best',
                                     'model-' + str(current_max_validation_index) + '.meta')
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
                    print("Epoch            : %d, loss: %f" % (i + 1, loss_val))
                    print(
                        "Current Max Index: %d, loss: %f" % (current_max_validation_index + 1, current_min_loss))
                    print("total Max Index  : %d, loss: %f" % (total_max_validation_index + 1, total_min_loss))
                    print(
                        "=================================================================================================")
                    # print("")

                    f.write(
                        "===============================================================================\n")
                    f.write(
                        "Epoch               : %d, loss: %f\n" % (
                        i + 1, loss_val))
                    f.write("Current Max Index: %d, loss: %f\n" % (
                    current_max_validation_index + 1, current_min_loss))
                    f.write("total Max Index: %d, loss: %f\n" % (total_max_validation_index + 1, total_min_loss))
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

                    lr = lr_init * (epochs-(i+1))/epochs
                    print('Learning rate is changed to ', lr)

                    print("")
                    if total_max_validation_index==(i-3000):
                        break

                # if training_loss <= 0.01:
                #     break

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