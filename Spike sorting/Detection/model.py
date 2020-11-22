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

def build_model_ResNet(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name = ''):
    x_data = x
    x_data = CNN_layer(x_data, channel_list[0], kernel_list[0], stride_list[0], reuse_ = re, name = name+'Stem', padding = 'SAME')

    for i in range(1, len(channel_list)): # Residual block
        with tf.variable_scope("Residual_Block-" + str(i)):
            x_data_1 = CNN_layer(x_data, x_data.shape[-1], [1, 1], [1, 1], reuse_=re, name=name +'1x1'+str(i)+'-bf', padding='SAME')
            x_data_2 = CNN_layer(x_data_1, x_data.shape[-1], kernel_list[i], stride_list[i], reuse_=re, name=name + str(i),
                               padding='SAME')
            x_data_3 = CNN_layer(x_data_2, channel_list[i], [1, 1], [1, 1], reuse_=re, name=name +'1x1'+str(i)+'-af', padding='SAME')
            if x_data.shape[-1] != x_data_3.shape[-1]:
                x_data = CNN_layer(x_data, x_data_3.shape[-1], [1, 1], [1, 1], reuse_=re, name=name +'1x1'+str(i)+'-ch', padding='SAME')
            x_data = x_data + x_data_3
        if i % 2 == 1 and i != 0:
            with tf.variable_scope('Dropout-'+str(int(i/2))):
                # x_data = tf.layers.batch_normalization(x_data, name = 'BatchNorm-'+name+str(int(i/2)))
                x_data = tf.layers.dropout(x_data, rate = keep_prob, name = 'Dropout-'+ name+str(int(i/2)))

    x_data = tf.layers.average_pooling2d(x_data, pool_size=[x_data.shape[1], x_data.shape[2]], strides=[1, 1],
                                         padding='VALID', name='GAV')
    x_data = tf.layers.flatten(x_data)
    for i in range(len(dense_list) - 1):
        x_data = tf.layers.dense(x_data, units=dense_list[i], activation=tf.nn.elu, reuse=re,
                                 name='FC_layer-' + name + str(int(i)))

    x_data = tf.layers.dense(x_data, units=dense_list[-1], activation=tf.nn.elu, reuse=re, name='Sigmoid-' + name)
    x_data = tf.nn.softmax(x_data, name='output')

    return x_data



def build_model_CNN_modi(x, keep_prob, channel_list, kernel_list, stride_list, dense_list, maxpool_list, re, name = ''):
    for i in range(len(channel_list)):
        layer = CNN_layer(x[:, :, :, 0:1], channel_list[i], kernel_list[i], stride_list[i], reuse_ = re, name = 'signal'+str(i), padding = 'SAME')
        layer_SNEO = CNN_layer(x[:, :, :, 1:2], channel_list[i], kernel_list[i], stride_list[i], reuse_ = re, name = 'SNEO'+str(i), padding = 'SAME')
    layer = tf.layers.average_pooling2d(layer, pool_size = [layer.shape[1], layer.shape[2]], strides = [1, 1], padding = 'VALID', name = 'GAV')
    layer_SNEO = tf.layers.average_pooling2d(layer_SNEO, pool_size=[layer_SNEO.shape[1], layer_SNEO.shape[2]], strides=[1, 1],
                                        padding='VALID', name='GAV')
    layer = tf.layers.flatten(layer)
    layer_SNEO = tf.layers.flatten(layer_SNEO)
    layer = tf.concat((layer, layer_SNEO), axis = -1)
    for i in range(len(dense_list)-1):
        layer = tf.layers.dense(layer, units = dense_list[i], activation = tf.nn.elu, reuse = re, name = 'FC_layer-'+name+str(int(i)))
        if i % 1 == 0 and i != 0:
            with tf.variable_scope('Dropout-' + str(int(i / 2))):
                # x_data = tf.layers.batch_normalization(x_data, name = 'BatchNorm-'+name+str(int(i/2)))
                layer = tf.layers.dropout(layer, rate=keep_prob, name='Dropout-' + name + str(int(i / 2)))

    layer = tf.layers.dense(layer, units=dense_list[-1], activation=tf.nn.elu, reuse=re, name='Sigmoid-' + name)
    layer = tf.nn.softmax(layer, name='output')

    return layer