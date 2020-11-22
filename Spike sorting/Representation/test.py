import numpy as np
import tensorflow as tf
import argparse
import os
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Spike sorting")
parser.add_argument("-t", "--time", type = str, default = "2020_10_24_20_30")
parser.add_argument("-g", "--gpu", type = int, default = 0)

args = parser.parse_args()
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)

data_path = '/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week'
save_path = '/media/sion/D/Data'
load_path = os.path.join(save_path, 'checkpoint/Spike_sorting/representation/'+args.time)
f = open(load_path + '/log.txt')
lines = f.readlines()
file_num = lines[-2].split(':')[1]
file_num = file_num.split(',')[0]
file_num = file_num.split(' ')[1]
f.close()
restore_path = load_path + '/model-' + str(int(file_num) - 1)

with tf.variable_scope('Load_Data'):
    test_dict = np.load(os.path.join(data_path, 'npy/test.npy'), allow_pickle=True).item()
    test_idx = np.load(os.path.join(data_path, 'npy/test_idx.npy'), allow_pickle=True).item()
    test_key = list(test_dict.keys())
    print(test_key)
    test_file = 2
    print("Test file: ", test_key[test_file])
    data_test = []
    label_test = []
    for i in range(len(test_key)):
        if i == test_file:
            data_test_ = test_dict[test_key[i]]['signal']
            label_test_ = test_dict[test_key[i]]['label']
            idx = test_idx[test_key[i]]
            for j in range(1, len(data_test_)):
                data_test.append(data_test_[j][idx[j]])
                label_test.append(label_test_[j][idx[j]])
    data_test = np.concatenate(data_test, axis=0)
    data_test = data_test[:, :, :, 0:1]
    label_test = np.concatenate(label_test, axis = 0)
    print(np.unique(np.argmax(label_test, axis = 1), return_counts = True))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(load_path, 'model-'+str(int(file_num)-1)+'.meta'))
    new_saver.restore(sess, restore_path)
    # parameter = tf.trainable_variables()
    # para = sess.run(parameter)
    # for i in range(len(para)):
    #     print(para[i].name)
    x = tf.get_default_graph().get_tensor_by_name("Graph_build/Placeholder/Placeholder:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("Graph_build/Placeholder/Placeholder_3:0")
    output = tf.get_default_graph().get_tensor_by_name("Graph_build/Model/output:0")

    batch = 1000
    run = int(data_test.shape[0]/batch)
    if data_test.shape[0]%batch != 0:
        run += 1
    features = []
    for i in range(run):
        data = data_test[batch*i:batch*(i+1)]
        label = label_test[batch*i:batch*(i+1)]
        feature = sess.run(output, feed_dict={x: data[:, :, :, 0:1], keep_prob: 1.0})
        features.append(feature)

    features = np.concatenate(features, axis = 0) # [N, F]
    color = ['blue', 'red', 'black', 'gray', 'gold', 'm', 'pink', 'chartreuse', 'deepskyblue', 'olive', 'brown',
             'chocolate', 'orange']
    color_ = []
    for i in range(label_test.shape[0]):
        color_.append(color[int(np.argmax(label_test[i:i + 1], axis=1))])
    print("TSNE is running")
    tsne = TSNE(n_components=2, metric='correlation', init='random', method='exact', n_iter=1000,
                perplexity=30)
    tsne_ = tsne.fit_transform(features)
    print("TSNE is finishied")
    xf = tsne_[:, 0]
    yf = tsne_[:, 1]
    plt.scatter(xf, yf, s=2.0, c=color_, label=color_)
    plt.show()