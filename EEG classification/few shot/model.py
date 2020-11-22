import sys
sys.path.append('/home/sion/code')
import Code_total as ct
import tensorflow as tf
import numpy as np


def log_distance(x, margin):
    output = margin-tf.math.log(x)

    return output

def euclidean_dis(x1, x2):
    x1 = tf.layers.flatten(x1)
    x2 = tf.layers.flatten(x2)
    output = tf.sqrt(tf.reduce_sum(tf.pow(x1-x2, 2), 1))

    return output

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

    # x_data = tf.layers.flatten(x_data)
    # for i in range(len(dense_list)-1):
    #     x_data = tf.layers.dense(x_data, units = dense_list[i], activation = tf.nn.elu, reuse = re, name = 'FC_layer-'+name+str(int(i)))
    #
    # x_data = tf.layers.dense(x_data, units = dense_list[-1], activation = tf.nn.sigmoid, reuse = re, name = 'Sigmoid-'+name)

    return x_data


def build_graph_GAT(x, feature, multi_head, A, reuse):
    # x = [N, feature, Node]
    reuse_ = reuse
    with tf.variable_scope("GAT", reuse=reuse):
        Adj = tf.get_variable("Adj", initializer=A, dtype=tf.float32, trainable=False)
    H = tf.transpose(x, [0, 2, 1])
    for i in range(len(feature)):
        H = tf.layers.conv1d(H, feature[i], 1, activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='w_' + str(i), reuse=reuse)
        Att_list = []
        for j in range(multi_head[i]):
            # row_att = []
            # for row in range(H.shape[1]):
            #     col_att = []
            #     for col in range(H.shape[1]):
            #         H_Att = tf.concat((H[:, row], H[:, col]), axis = -1)
            #         col_att.append(tf.layers.dense(H_Att, units = 1, activation = tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                                    name = 'Att_'+str(i)+'_'+str(j), reuse = reuse))
            #         reuse = True
            #     row_att.append(tf.expand_dims(tf.concat(col_att, axis = -1), axis = 1))
            # reuse = reuse_
            # Att = tf.concat(row_att, axis = 1)

            # H_Att = tf.layers.flatten(H)
            # Att_ij = tf.layers.dense(H_Att, units=H.shape[1], activation=tf.nn.leaky_relu,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                          name = 'aij_'+str(i)+'_'+str(j), reuse = reuse)
            # Att_ji = tf.layers.dense(H_Att, units=H.shape[1], activation=tf.nn.leaky_relu,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                          name = 'aji_'+str(i)+'_'+str(j), reuse = reuse)
            # Att_ij = tf.expand_dims(Att_ij, -1)
            # Att_ji = tf.expand_dims(Att_ji, -1)

            Att_ij = tf.layers.conv1d(H, H.shape[1], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='aij_' + str(i) + '_' + str(j), reuse=reuse)
            Att_ji = tf.layers.conv1d(H, H.shape[1], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='aji_' + str(i) + '_' + str(j), reuse=reuse)

            Att = Att_ij + tf.transpose(Att_ji, [0, 2, 1])
            Att = tf.nn.leaky_relu(Att * Adj)
            # Att = tf.nn.softmax(tf.nn.leaky_relu(Att), 2)
            Attsum = tf.expand_dims(tf.reduce_sum(Att, -1), -1)
            Att = Att / Attsum
            Att_list.append(Att)  # Att*Adj

        H_Att_list = []
        for j in range(multi_head[i]):
            H_Att = tf.matmul(Att_list[j], H)
            H_Att = tf.contrib.layers.bias_add(H_Att)
            H_Att_list.append(H_Att)
        H = tf.nn.leaky_relu(tf.concat(H_Att_list, axis=-1))

    H = tf.transpose(H, [0, 2, 1])

    return H, Att_list


def build_graph_GCN(x, feature, A, re):
    size = x.get_shape().as_list()  # N, F, #node
    # A_init = np.float32(np.ones((size[2], size[2])))
    # D_init = np.float32(np.zeros((size[2], size[2])))
    # for i in range(D_init.shape[0]):
    #     D_init[i, i] = np.sum(A_init[i])
    # A = tf.get_variable("Adjacency",  initializer = A)
    # D = tf.get_variable("Degree",  initializer= D_init, trainable = False)
    # D_inv = tf.linalg.inv(D)**0.5
    w_list = []
    # b_list = []
    with tf.variable_scope("GCN_", reuse=re):
        for i in range(len(feature)):
            if i == 0:
                w = tf.get_variable("w_" + str(i), shape=[size[1], feature[i]],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            else:
                w = tf.get_variable("w_" + str(i), shape=[feature[i - 1], feature[i]],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # b = tf.get_variable("b_" + str(i), shape=[feature[i]], dtype = tf.float32)
            w_list.append(w)
            # b_list.append(b)

        H = tf.transpose(x, [0, 2, 1])
        # D = tf.reduce_sum(A, 2)
        # D_diag = tf.linalg.diag(D)
        # D_inv = tf.linalg.inv(D_diag)
        # D_inv = D_inv ** 0.5
        # norm_A = tf.matmul(tf.matmul(D_inv, A), D_inv)
        for i in range(len(feature)):
            H = tf.matmul(tf.contrib.layers.bias_add(H, initializer=tf.contrib.layers.xavier_initializer()), w_list[i])
            H = tf.matmul(A, tf.nn.elu(H))

    return H


def build_graph_GAT_conv(x, kernel, channel, multi_head, Adj):
    size = x.get_shape().as_list()  # N, F, #node # channel
    Amat = {}
    H = tf.transpose(x, [0, 2, 1, 3])
    for i in range(len(kernel)):
        w = tf.get_variable("w_" + str(i), shape=[1, kernel[i], H.get_shape().as_list()[-1], channel[i]],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        H = tf.nn.conv2d(H, w, [1, 1], padding='VALID')
        H = tf.transpose(tf.nn.elu(H), [0, 3, 1, 2])  # N, channel, F, #Node
        Att_list = []
        for j in range(multi_head[i]):
            a_ij = tf.get_variable("aij_" + str(i) + "_" + str(j), shape=[channel[i], H.get_shape().as_list()[-1], 1],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            a_ji = tf.get_variable("aji_" + str(i) + "_" + str(j), shape=[channel[i], H.get_shape().as_list()[-1], 1],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            Att_ij = tf.matmul(H, a_ij)
            Att_ji = tf.matmul(H, a_ji)
            Att = Att_ij + tf.transpose(Att_ji, [0, 1, 3, 2])
            Att = tf.nn.leaky_relu(Att * Adj)
            Att = tf.nn.softmax(tf.contrib.layers.bias_add(Att, initializer=tf.contrib.layers.xavier_initializer()))
            if j == 0:
                H_Att = tf.matmul(Att, H)
            else:
                H_Att = tf.concat((H_Att, H), axis=1)
            Att_list.append(Att)
        H = tf.nn.elu(tf.transpose(H_Att, [0, 2, 3, 1]))
        Amat[str(i)] = Att_list

    return H, Att_list


def RelationNet(support, query, model, kp, type):
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
            if i == 0 and type == '_att':
                output = tf.expand_dims(sup_output, axis=-1)
                output_ = tf.expand_dims(sup_output_, axis=-1)
            if i != 0 and type == '_att':
                output = tf.concat((output, tf.expand_dims(sup_output, axis=-1)), axis=-1)
                output_ = tf.concat((output_, tf.expand_dims(sup_output_, axis=-1)), axis=-1)

            if i == 0 and type == '_dis':
                output = tf.expand_dims(sup_output, axis=-1)
                output_ = tf.expand_dims(sup_output_, axis=-1)
            if i != 0 and type == '_dis':
                output = tf.concat((output, tf.expand_dims(sup_output, axis=-1)), axis=-1)
                output_ = tf.concat((output_, tf.expand_dims(sup_output_, axis=-1)), axis=-1)

            if i == 0 and type == '_satt':
                output = tf.expand_dims(sup_output, axis=-1)
                output_ = tf.expand_dims(sup_output_, axis=-1)
            if i != 0 and type == '_satt':
                output = tf.concat((output, tf.expand_dims(sup_output, axis=-1)), axis=-1)
                output_ = tf.concat((output_, tf.expand_dims(sup_output_, axis=-1)), axis=-1)

            if i == 0 and type == '_gat':
                output = sup_output
                output_ = sup_output_
            if i != 0 and type == '_gat':
                output = tf.concat((output, sup_output), axis=2)
                output_ = tf.concat((output_, sup_output_), axis=2)

            if i == 0 and type == '_gcn':
                output = sup_output
                output_ = sup_output_
            if i != 0 and type == '_gcn':
                output = tf.concat((output, sup_output), axis=2)
                output_ = tf.concat((output_, sup_output_), axis=2)

        if model == 'EEGNet':
            que_output = build_model_EEGNet(query[:, 0], kp, re)
        if model == 'ShallowNet':
            que_output = build_model_ShallowNet(query[:, 0], kp, re)
        if model == 'DeepconvNet':
            que_output = build_model_DeepconvNet(query[:, 0], kp, re)
        if model == 'HS_CNN':
            que_output = build_model_HS_CNN(query[:, 0], kp, re)

    if type == '_satt':
        with tf.variable_scope('self_attention'):
            reuse_ = False
            query_vector = tf.layers.max_pooling2d(que_output, pool_size = [que_output.shape[1], que_output.shape[2]],
                                                   strides = [1, 1], padding = 'VALID')
            query_vector = tf.layers.flatten(query_vector)
            query_vector = tf.layers.dense(query_vector, units = query_vector.shape[1], activation= tf.nn.elu,
                                           kernel_initializer= tf.contrib.layers.xavier_initializer(), reuse=reuse_, name = 'query_key')
            reuse_ = True
            for N in range(few):
                print("Before", output.get_shape().as_list())
                key_vector = tf.layers.max_pooling2d(output[:, :, :, :, N], pool_size=[output.shape[1], output.shape[2]],
                                                       strides=[1, 1], padding='VALID')
                print("After", key_vector.get_shape().as_list())
                key_vector = tf.layers.flatten(key_vector)
                key_vector = tf.layers.dense(key_vector, units=key_vector.shape[1], activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_, name = 'query_key')

                satt_ = tf.matmul(tf.expand_dims(query_vector, axis=2), tf.expand_dims(key_vector, axis=1))
                satt_ = tf.linalg.diag_part(satt_)
                satt_ = satt_ / tf.math.sqrt(tf.cast(query_vector.shape[1], tf.float32))
                satt_ = tf.expand_dims(satt_, axis = -1)
                if N == 0:
                    satt = satt_
                else:
                    satt = tf.concat((satt, satt_), axis = -1)
                reuse_ = True

            satt = tf.nn.softmax(satt, axis = -1)
            satt = tf.reshape(satt, [-1, 1, 1, satt.shape[1], satt.shape[2]])
            rep_feature = output * satt
            rep_feature = tf.reduce_sum(rep_feature, axis = -1)

            for N in range(few):
                key_vector = tf.layers.max_pooling2d(output_[:, :, :, :, N], pool_size=[output_.shape[1], output_.shape[2]],
                                                     strides=[1, 1], padding='VALID')
                key_vector = tf.layers.flatten(key_vector)
                key_vector = tf.layers.dense(key_vector, units=key_vector.shape[1], activation=tf.nn.elu,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_, name = 'query_key')

                satt_ = tf.matmul(tf.expand_dims(query_vector, axis=2), tf.expand_dims(key_vector, axis=1))
                satt_ = tf.linalg.diag_part(satt_)
                satt_ = satt_ / tf.math.sqrt(tf.cast(query_vector.shape[1], tf.float32))
                satt_ = tf.expand_dims(satt_, axis=-1)
                if N == 0:
                    satt = satt_
                else:
                    satt = tf.concat((satt, satt_), axis=-1)
                reuse_ = True
            satt = tf.nn.softmax(satt, axis=-1)
            satt = tf.reshape(satt, [-1, 1, 1, satt.shape[1], satt.shape[2]])
            rep_feature_ = output_ * satt
            rep_feature_ = tf.reduce_sum(rep_feature_, axis=-1)

            # rep_feature = tf.concat((rep_feature, que_output), axis = -1)
            # rep_feature_ = tf.concat((rep_feature_, que_output), axis = -1)

    if type == '_gcn':
        with tf.variable_scope('GCN'):
            att_input = tf.concat((output, que_output), axis=2)
            re_att = False
            emb = []
            for node in range(few):
                emb.append(
                    build_model_CNN(output[:, :, node:node + 1, :], kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                    [[1, 1], [1, 1]], [256, 100, 1],
                                    np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention'))
                re_att = True

            emb.append(build_model_CNN(que_output[:, :, :, :], kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                       [[1, 1], [1, 1]], [256, 100, 1],
                                       np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention'))
            adj = []
            for row in range(few + 1):
                Adj = []
                for col in range(few + 1):
                    Adj.append(tf.norm(emb[row] - emb[col], ord='euclidean'))
                    import pdb
                    pdb.set_trace()

                Adjsum = tf.reduce_sum(Adj, -1)
                Adj = (Adjsum - Adj) / Adjsum
                adj.append(Adj)

            emb = []
            for node in range(few):
                emb.append(
                    build_model_CNN(output_[:, :, node:node + 1, :], kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                    [[1, 1], [1, 1]], [256, 100, 1],
                                    np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention'))
            emb.append(build_model_CNN(que_output[:, :, :, :], kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                       [[1, 1], [1, 1]], [256, 100, 1],
                                       np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention'))
            adj_ = []
            for row in range(few + 1):
                for col in range(few + 1):
                    if col == 0:
                        Adj = tf.norm(emb[row] - emb[col], ord='euclidean')
                    else:
                        Adj = tf.concat((Adj, tf.norm(emb[row] - emb[col], ord='euclidean')), axis=-1)

                Adjsum = tf.reduce_sum(Adj, -1)
                Adj = (Adjsum - Adj) / Adjsum
                adj_.append(Adj)

            att_input = tf.concat((output, que_output), axis=2)
            adj = []
            for row in range(few+1):
                for col in range(few+1):
                    sup_que = tf.concat((att_input[:, :, row:row+1, :], att_input[:, :, col:col+1, :]), axis = -1)
                    if col == 0:
                        Adj = ct.build_model_CNN(sup_que, kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                                                  [[1, 1], [1, 1]], [256, 100, 1],
                                                                  np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention')
                    else:
                        Adj = tf.concat((Adj, ct.build_model_CNN(sup_que, kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                                                  [[1, 1], [1, 1]], [256, 100, 1],
                                                                  np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention')), axis = -1)
                    re_att = True
                adj.append(tf.expand_dims(Adj, axis = 1))
            Adj = tf.concat(adj, axis = 1)
            Adjsum = tf.expand_dims(tf.reduce_sum(Adj, 2), axis = 2)
            Adj = Adj/Adjsum
            print("Adj", Adj.get_shape().as_list())

            att_input = tf.concat((output_, que_output), axis=2)
            adj = []
            for row in range(few + 1):
                for col in range(few + 1):
                    sup_que = tf.concat((att_input[:, :, row:row + 1, :], att_input[:, :, col:col + 1, :]), axis=-1)
                    if col == 0:
                        Adj_ = ct.build_model_CNN(sup_que, kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                                 [[1, 1], [1, 1]], [256, 100, 1],
                                                 np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention')
                    else:
                        Adj_ = tf.concat(
                            (Adj_, ct.build_model_CNN(sup_que, kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                                     [[1, 1], [1, 1]], [256, 100, 1],
                                                     np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention')), axis=-1)
                    re_att = True
                adj.append(tf.expand_dims(Adj_, axis = 1))
            Adj_ = tf.concat(adj, axis=1)
            Adjsum_ = tf.expand_dims(tf.reduce_sum(Adj_, 2), axis = 2)
            Adj_ = Adj_/Adjsum_

            output_shape = output.get_shape().as_list()
            output = tf.layers.average_pooling2d(output, pool_size=[output_shape[1], 1], strides=[1, 1],
                                                 padding='VALID', name='GCN_GAP_S')
            output_ = tf.layers.average_pooling2d(output_, pool_size=[output_shape[1], 1], strides=[1, 1],
                                                  padding='VALID', name='GCN_GAP_S_')
            que_output = tf.layers.average_pooling2d(que_output, pool_size=[output_shape[1], 1], strides=[1, 1],
                                                     padding='VALID', name='GCN_GAP_Q')
            graph = tf.concat((output, output_, que_output), axis=2)
            # graph_shape = graph.get_shape().as_list()
            # graph = tf.layers.conv2d(graph, graph.get_shape().as_list()[-1], [int(graph_shape[1] / 10), 1],
            #                          strides=[int(graph_shape[1] / 20), 1], padding='VALID')
            graph_shape = graph.get_shape().as_list()
            graph = tf.transpose(graph, [0, 1, 3, 2])
            graph_ = tf.concat((graph[:, :, :, :few], graph[:, :, :, -1:]), axis=-1)
            graph__ = graph[:, :, :, few:]
            graph_ = tf.reshape(graph_, [-1, graph_shape[1] * graph_shape[3], int(graph_shape[2] / 2) + 1])
            graph__ = tf.reshape(graph__, [-1, graph_shape[1] * graph_shape[3], int(graph_shape[2] / 2) + 1])
            graph_shape = graph_.get_shape().as_list()
            print("Graph input", graph_.get_shape().as_list())
            graph_ = build_graph_GCN(graph_, [int(graph_shape[1]), int(graph_shape[1])], Adj,
                                     False)  # output = N, Node, F
            graph__ = build_graph_GCN(graph__, [int(graph_shape[1]), int(graph_shape[1])], Adj_, True)
            graph_ = tf.transpose(graph_, [0, 2, 1])
            graph__ = tf.transpose(graph__, [0, 2, 1])
            graph_ = tf.reshape(graph_, [-1, output.shape[1], few + 1, output.shape[-1]])
            graph__ = tf.reshape(graph__, [-1, output.shape[1], few + 1, output.shape[-1]])

            weight_sum = tf.reshape(tf.reduce_sum(Adj[:, few, :-1], 1), [-1, 1, 1, 1])
            weight_sum_ = tf.reshape(tf.reduce_sum(Adj_[:, few, :-1], 1), [-1, 1, 1, 1])

            for adj in range(few):
                if adj == 0:
                    rep_feature = (output[:, :, adj:adj + 1, :] + graph_[:, :, adj:adj + 1, :]) * tf.reshape(
                        Adj[:, few, adj], [-1, 1, 1, 1]) / weight_sum
                else:
                    rep_feature = tf.add(rep_feature, (output[:, :, adj:adj + 1, :] + graph_[:, :, adj:adj + 1, :])
                                         * tf.reshape(Adj[:, few, adj], [-1, 1, 1, 1])) / weight_sum
            for adj in range(few):
                if adj == 0:
                    rep_feature_ = (output_[:, :, adj:adj + 1, :] + graph__[:, :, adj:adj + 1, :]) * tf.reshape(
                        Adj_[:, few, adj], [-1, 1, 1, 1]) / weight_sum_
                else:
                    rep_feature_ = tf.add(rep_feature_, (output_[:, :, adj:adj + 1, :] + graph__[:, :, adj:adj + 1, :])
                                          * tf.reshape(Adj_[:, few, adj], [-1, 1, 1, 1])) / weight_sum_

            rep_feature = tf.concat((rep_feature, que_output + graph_[:, :, -1:, :]), axis=-1)
            rep_feature_ = tf.concat((rep_feature_, que_output + graph__[:, :, -1:, :]), axis=-1)

    if type == '_gat':
        with tf.variable_scope('GAT'):
            graph = tf.concat((output, output_, que_output), axis=2)
            # graph_shape = graph.get_shape().as_list()
            # graph = tf.layers.conv2d(graph, graph.get_shape().as_list()[-1], [int(graph_shape[1] / 10), 1],
            #                            strides=[int(graph_shape[1] / 20), 1], padding='VALID')
            graph_shape = graph.get_shape().as_list()
            graph = tf.layers.average_pooling2d(graph, [graph_shape[1], 1], [1, 1])
            graph_shape = graph.get_shape().as_list()
            graph = tf.transpose(graph, [0, 1, 3, 2])  # gat
            graph_ = tf.concat((graph[:, :, :, :few], graph[:, :, :, -1:]), axis=-1)
            graph__ = graph[:, :, :, few:]
            graph_ = tf.reshape(graph_, [-1, graph_shape[1] * graph_shape[3], int(graph_shape[2] / 2) + 1])  # gat
            graph__ = tf.reshape(graph__, [-1, graph_shape[1] * graph_shape[3], int(graph_shape[2] / 2) + 1])  # gat
            print(graph_.get_shape().as_list(), "Graph input")
            graph_shape = graph_.get_shape().as_list()
            Adj = np.float32(np.zeros((few + 1, few + 1)))
            for adj in range(Adj.shape[0]):
                Adj[adj, adj] = 1
            Adj[-1, :] = 1
            Adj[:, -1] = 1
            # print(Adj)
            # test = np.float32(np.ones_like(Adj))
            # print(test*Adj)
            # import pdb
            # pdb.set_trace()

            graph_, Attention_ = build_graph_GAT(graph_, [int(graph_shape[1] / 2), int(graph_shape[1] / 8),
                                                          int(graph_shape[1] / 16), 1], [3, 3, 3, 1], Adj,
                                                 False)  # gat # N, F, node
            graph__, Attention__ = build_graph_GAT(graph__, [int(graph_shape[1] / 2), int(graph_shape[1] / 8),
                                                             int(graph_shape[1] / 16), 1], [3, 3, 3, 1], Adj,
                                                   True)  # gat # N, F, node

            # graph, Attention = build_graph_GAT_conv(graph, [int(graph.get_shape().as_list()[1]/4)], [int(graph.get_shape().as_list()[-1])], [1], Adj)
            # N, Node, F, Channel # conv

            Att = graph_  # gat
            Att_ = graph__  # gat
            Att__ = tf.concat((Attention_, Attention__), axis=-1)

            # Att = tf.nn.softmax(Attention[0][:, :, :, :few]) #conv
            # Att_ = tf.nn.softmax(Attention[0][:, :, :, few:-1]) #conv

            weight_sum = tf.reshape(tf.reduce_sum(graph_[:, 0, :few]), [-1, 1, 1, 1])
            weight_sum_ = tf.reshape(tf.reduce_sum(graph__[:, 0, :few]), [-1, 1, 1, 1])

            for adj in range(few):  # gat
                if adj == 0:
                    rep_feature = output[:, :, adj:adj + 1, :] * tf.reshape(Att[:, 0, adj], [-1, 1, 1, 1]) / weight_sum
                else:
                    rep_feature = tf.add(rep_feature, output[:, :, adj:adj + 1, :] * tf.reshape(Att[:, 0, adj],
                                                                                                [-1, 1, 1,
                                                                                                 1])) / weight_sum
            for adj in range(few):
                if adj == 0:
                    rep_feature_ = output_[:, :, adj:adj + 1, :] * tf.reshape(Att_[:, 0, adj],
                                                                              [-1, 1, 1, 1]) / weight_sum_
                else:
                    rep_feature_ = tf.add(rep_feature_, output_[:, :, adj:adj + 1, :] * tf.reshape(Att_[:, 0, adj],
                                                                                                   [-1, 1, 1,
                                                                                                    1])) / weight_sum_

            # for adj in range(few):
            #     if adj == 0:
            #         rep_feature = output[:, :, adj:adj+1, :]*tf.reshape(Att[:, :, 2*few, adj],[-1, 1, 1, Att.get_shape().as_list()[1]])
            #     else:
            #         rep_feature = tf.add(rep_feature, output[:, :, adj:adj+1, :]*tf.reshape(Att[:, :, 2*few, adj],[-1, 1, 1, Att.get_shape().as_list()[1]]))
            # for adj in range(few):
            #     if adj == 0:
            #         rep_feature_ = output_[:, :, adj:adj+1, :]*tf.reshape(Att_[:, :, 2*few, adj],[-1, 1, 1, Att_.get_shape().as_list()[1]])
            #     else:
            #         rep_feature_ = tf.add(rep_feature_, output_[:, :, adj:adj+1, :]*tf.reshape(Att_[:, :, 2*few, adj],[-1, 1, 1, Att_.get_shape().as_list()[1]]))

            rep_feature = tf.concat((rep_feature, que_output), axis=-1)
            rep_feature_ = tf.concat((rep_feature_, que_output), axis=-1)

    if type == '_att':
        with tf.variable_scope('Attention'):
            attention_score = {}
            attention_score_ = {}
            re_att = False
            for attention in range(few):
                sup_que = tf.concat((output[:, :, :, :, attention], que_output), axis=-1)
                attention_score_each = ct.build_model_CNN(sup_que, kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                                          [[1, 1], [1, 1]], [256, 100, 1],
                                                          np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention')
                re_att = True
                attention_score[str(attention)] = attention_score_each

            for attention in range(few):
                sup_que = tf.concat((output_[:, :, :, :, attention], que_output), axis=-1)
                attention_score_each = ct.build_model_CNN(sup_que, kp, [128, 256], np.asarray([[30, 1], [15, 1]]),
                                                          [[1, 1], [1, 1]], [256, 100, 1],
                                                          np.asarray([[0, 3, 1, 3, 1]]), re_att, 'Attention')
                attention_score_[str(attention)] = attention_score_each

            weight_key = list(attention_score.keys())
            for weight in range(len(weight_key)):
                if weight == 0:
                    weight_sum = attention_score[weight_key[weight]]
                else:
                    weight_sum += attention_score[weight_key[weight]]

            for weight in range(len(weight_key)):
                if weight == 0:
                    weight_sum_ = attention_score_[weight_key[weight]]
                else:
                    weight_sum_ += attention_score_[weight_key[weight]]

            weight_sum = tf.reshape(weight_sum, [-1, 1, 1, 1])
            weight_sum_ = tf.reshape(weight_sum_, [-1, 1, 1, 1])

            for weight in range(len(weight_key)):
                if weight == 0:
                    rep_feature = output[:, :, :, :, weight] * tf.reshape(attention_score[weight_key[weight]],
                                                                          [-1, 1, 1, 1]) / weight_sum
                else:
                    rep_feature = rep_feature + (
                            output[:, :, :, :, weight] * tf.reshape(attention_score[weight_key[weight]],
                                                                    [-1, 1, 1, 1]) / weight_sum)

            for weight in range(len(weight_key)):
                if weight == 0:
                    rep_feature_ = output_[:, :, :, :, weight] * tf.reshape(attention_score_[weight_key[weight]],
                                                                            [-1, 1, 1, 1]) / weight_sum_
                else:
                    rep_feature_ = rep_feature_ + (
                            output_[:, :, :, :, weight] * tf.reshape(attention_score_[weight_key[weight]],
                                                                     [-1, 1, 1, 1]) / weight_sum_)

            rep_feature = tf.concat((rep_feature, que_output), axis=-1)
            rep_feature_ = tf.concat((rep_feature_, que_output), axis=-1)
    if type == '_dis':
        with tf.variable_scope('DISTANCE'):
            attention_score = []
            attention_score_ = []
            # test = []
            for w in range(output.shape[-1]):
                # test.append(euclidean_dis(que_output, output[:, :, :, :, w]))
                attention_score.append(log_distance(euclidean_dis(que_output, output[:, :, :, :, w]), 5))
                attention_score_.append(log_distance(euclidean_dis(que_output, output_[:, :, :, :, w]), 5))
            for weight in range(len(attention_score)):
                if weight == 0:
                    weight_sum = attention_score[weight]
                else:
                    weight_sum += attention_score[weight]

            for weight in range(len(attention_score)):
                if weight == 0:
                    weight_sum_ = attention_score_[weight]
                else:
                    weight_sum_ += attention_score_[weight]

            weight_sum = tf.reshape(weight_sum, [-1, 1, 1, 1])
            weight_sum_ = tf.reshape(weight_sum_, [-1, 1, 1, 1])

            for weight in range(len(attention_score)):
                if weight == 0:
                    rep_feature = output[:, :, :, :, weight] * tf.reshape(attention_score[weight],
                                                                          [-1, 1, 1, 1]) / weight_sum
                else:
                    rep_feature = rep_feature + (
                            output[:, :, :, :, weight] * tf.reshape(attention_score[weight],
                                                                    [-1, 1, 1, 1]) / weight_sum)

            for weight in range(len(attention_score)):
                if weight == 0:
                    rep_feature_ = output_[:, :, :, :, weight] * tf.reshape(attention_score_[weight],
                                                                            [-1, 1, 1, 1]) / weight_sum_
                else:
                    rep_feature_ = rep_feature_ + (
                            output_[:, :, :, :, weight] * tf.reshape(attention_score_[weight],
                                                                     [-1, 1, 1, 1]) / weight_sum_)

            rep_feature = tf.concat((rep_feature, que_output), axis=-1)
            rep_feature_ = tf.concat((rep_feature_, que_output), axis=-1)

    with tf.variable_scope('Relation'):
        layer = build_model_CNN(rep_feature, kp, [128, 256], np.asarray([[30, 1], [15, 1]]), [[1, 1], [1, 1]],
                                [256, 100, 1],
                                np.asarray([[0, 3, 1, 3, 1]]), False, 'Relation')
        layer_ = build_model_CNN(rep_feature_, kp, [128, 256], np.asarray([[30, 1], [15, 1]]), [[1, 1], [1, 1]],
                                 [256, 100, 1],
                                 np.asarray([[0, 3, 1, 3, 1]]), True, 'Relation')
        layer__ = build_model_CNN(que_output, kp, [128, 256], np.asarray([[30, 1], [15, 1]]), [[1, 1], [1, 1]],
                                  [256, 100, 1],
                                  np.asarray([[0, 3, 1, 3, 1]]), True, 'Relation')
        layer = tf.layers.flatten(layer)
        layer_ = tf.layers.flatten(layer_)
        layer__ = tf.layers.flatten(layer__)
        # layer = tf.concat((tf.layers.flatten(layer), graph[:, :, -1]), axis = -1)
        # layer_ = tf.concat((tf.layers.flatten(layer_), graph[:, :, -1]), axis = -1)

        dense_list = [512, 256, 64, 1]
        for dense in range(len(dense_list) - 1):
            if dense == 0:
                layer = tf.concat((layer, layer__), axis=-1)
                layer_ = tf.concat((layer_, layer__), axis=-1)
            re = False
            layer = tf.layers.dense(layer, units=dense_list[dense], activation=tf.nn.elu, reuse=re,
                                    name='FC_layer-' + str(dense))
            re = True
            layer_ = tf.layers.dense(layer_, units=dense_list[dense], activation=tf.nn.elu, reuse=re,
                                     name='FC_layer-' + str(dense))

        layer = tf.layers.dense(layer, units=dense_list[-1], activation=tf.nn.sigmoid, reuse=False,
                                name='output')
        layer_ = tf.layers.dense(layer_, units=dense_list[-1], activation=tf.nn.sigmoid, reuse=True,
                                 name='output')
        score = tf.concat((layer, layer_), axis=-1)

    if type == '_gat':
        #     with tf.variable_scope('Node_Classification'):
        #         # graph = tf.transpose(graph, [0, 2, 1, 3]) #conv
        #         graph_ = tf.expand_dims(graph_, axis = -1) # N, F, Node, 1
        #         graph__ = tf.expand_dims(graph__, axis=-1)  # N, F, Node, 1
        #         # graph_shape = graph.get_shape().as_list()
        #         # graph = tf.layers.average_pooling2d(graph, pool_size = [graph_shape[1], 1],strides = [graph_shape[1], 1],
        #         #                                     padding = 'VALID', name = 'graph_global_avg')
        #         node_output = []
        #         node_output_ = []
        #         re = False
        #         node_feature = tf.layers.flatten(graph_[:, :, -1, :])
        #         node_feature_ = tf.layers.flatten(graph__[:, :, -1, :])
        #         for dense in range(len(dense_list)-1):
        #             layer = tf.layers.dense(node_feature, units = dense_list[dense],
        #                                     activation = tf.nn.elu, reuse = re, name = 'Node-FC-'+str(dense))
        #             layer_ = tf.layers.dense(node_feature_, units=dense_list[dense],
        #                                     activation=tf.nn.elu, reuse=True, name='Node-FC-' + str(dense))
        #         layer = tf.layers.dense(layer, units = dense_list[-1], activation = tf.nn.sigmoid, reuse = re, name = 'Node-output')
        #         layer_ = tf.layers.dense(layer_, units=dense_list[-1], activation=tf.nn.sigmoid, reuse=True, name='Node-output')
        #         node_output.append(layer)
        #         node_output_.append(layer_)
        #
        #         node = []
        #         node = node + node_output[:-1] + node_output_[:-1] + node_output[-1:] + node_output_[-1:]
        #         node_output = tf.concat(node, axis = -1)
        #
        #     return score, Att, Att_, node_output, Att__
        return score, Att, Att_, Att__
    if type == '_gcn':
        return score, Adj, Adj_
    if type == '_satt':
        return score
    if type == '_att' or type == '_dis':
        return score, attention_score, attention_score_
