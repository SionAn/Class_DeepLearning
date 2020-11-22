import tensorflow as tf

def CNN_layer(input, channel, kernel, stride, reuse_ = False, name = ' ', padding = 'valid'):
    with tf.variable_scope('Conv-'+name):
        output = tf.layers.conv2d(inputs = input, filters = channel, kernel_size = kernel, \
                                  strides = stride, padding = padding, activation = tf.nn.elu, \
                                  reuse=reuse_, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
                                # tf.random_normal_initializer(stddev=0.00001, mean=0),

        return output

def build_model_CNN(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name = ''):
    x_data = x
    keep_prob = keep_prob
    pool_count = 0
    for i in range(len(channel_list)):
        x_data = CNN_layer(x_data, channel_list[i], kernel_list[i], stride_list[i], reuse_ = re, name = name+str(i), padding = 'SAME')
        if i % 2 == 1 and i != 0:
            with tf.variable_scope('Dropout-'+str(int(i/2))):
                # x_data = tf.layers.batch_normalization(x_data, name = 'BatchNorm-'+name+str(int(i/2)))
                x_data = tf.layers.dropout(x_data, rate = keep_prob, name = 'Dropout-'+ name+str(int(i/2)))
        if i in maxpool_list[::, 0]:
            x_data = tf.layers.average_pooling2d(x_data, pool_size = maxpool_list[pool_count, 1:3],
                                             strides = maxpool_list[pool_count, 3:5], padding = 'SAME',
                                             name = 'avg-'+name+str(int(i/2)))
            pool_count += 1

    x_data = tf.layers.average_pooling2d(x_data, pool_size = [x_data.shape[1], x_data.shape[2]], strides = [1, 1], padding = 'VALID', name = 'GAV')
    x_data = tf.layers.flatten(x_data)
    for i in range(len(dense_list)-1):
        x_data = tf.layers.dense(x_data, units = dense_list[i], activation = tf.nn.elu, reuse = re, name = 'FC_layer-'+name+str(int(i)))

    x_data = tf.layers.dense(x_data, units = dense_list[-1], activation = tf.nn.elu, reuse = re, name = 'Sigmoid-'+name)
    x_data = tf.nn.softmax(x_data, name = 'output')

    return x_data

def build_model_representation(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name = ''):
    query = build_model_CNN(x[:, 0], keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list,
                            re, name=name+'query')
    re_ = re
    with tf.variable_scope('momentum'):
        key_list = []
        for i in range(1, x.shape[1]):
            key = build_model_CNN(x[:, i], keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list,
                                    re_, name=name+'key')
            key = tf.stop_gradient(key)
            key_list.append(key)
            re_ = True

    return query, key_list

def build_model_query(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name = ''):
    query = build_model_CNN(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list,
                            re, name=name+'query')

    return query

def build_model_key(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name = ''):
    with tf.variable_scope('momentum'):
        key = build_model_CNN(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list,
                                re, name=name+'query')
        key = tf.stop_gradient(key)

    return key