import graph_build as graph
import numpy as np
import tensorflow as tf
import argparse
import os
import warnings
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser(description="Spike sorting")
parser.add_argument("-t", "--time", type = str, default = "2020_10_23_13_46")
parser.add_argument("-g", "--gpu", type = int, default = 0)
parser.add_argument("-c", "--gather", type = bool, default = False, help = 'gather data which prediction same as label from train, validation, testset')

args = parser.parse_args()
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
time = args.time
time = time[:4]+'_'+time[4:6]+'_'+time[6:8]+'_'+time[8:10]+'_'+time[10:]

data_path = '/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week'
save_path = '/media/sion/D/Data'
load_path = os.path.join(save_path, 'checkpoint/Spike_sorting/detection/'+time)
f = open(load_path + '/log.txt')
lines = f.readlines()
file_num = lines[-2].split(':')[1]
file_num = file_num.split(',')[0]
file_num = file_num.split(' ')[1]
f.close()
restore_path = load_path + '/model-' + str(int(file_num) - 1)

with tf.variable_scope('Load_Data'):
    test_dict = np.load(os.path.join(data_path, 'npy/test.npy'), allow_pickle=True).item()
    test_key = list(test_dict.keys())
    data_test = []
    label_test = []
    label_cluster = []
    for i in range(len(test_key)):
        data_test.append(np.concatenate(test_dict[test_key[i]]['signal'], axis=0))
        label_c = np.concatenate(test_dict[test_key[i]]['label'], axis=0)
        label = np.zeros((label_c.shape[0], 2))
        for j in range(label_c.shape[0]):
            if label_c[j, 0] == 1.0:
                label[j, 0] = 1
            else:
                label[j, 1] = 1
        label_test.append(label)
        label_cluster.append(label_c)
    data_test = np.concatenate(data_test, axis=0)
    label_test = np.concatenate(label_test, axis=0)

    # aug = data_test[0, :, 0, 0] + np.random.normal(scale = 0.03, size = data_test[0, :, 0, 0].shape)
    # minimum = np.min(aug)
    # maximum = np.max(aug)
    # aug = (aug-minimum)/(maximum-minimum)
    # import matplotlib.pyplot as plt
    # plt.subplot(2, 1, 1)
    # plt.plot(data_test[0, :, 0, 0])
    # plt.subplot(2, 1, 2)
    # plt.plot(aug)
    # plt.show()
    # assert False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(load_path, 'model-'+str(int(file_num)-1)+'.meta'))
    new_saver.restore(sess, restore_path)
    # parameter = tf.trainable_variables()
    # para = sess.run(parameter)
    # for i in range(len(para)):
    #     print(para[i].shape)
    x = tf.get_default_graph().get_tensor_by_name("Graph_build/Placeholder/Placeholder:0")
    y = tf.get_default_graph().get_tensor_by_name("Graph_build/Placeholder/Placeholder_1:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("Graph_build/Placeholder/Placeholder_2:0")
    output = tf.get_default_graph().get_tensor_by_name("Graph_build/Model/output:0")

    prediction, answer, correct, accuracy = graph.class_accuracy(output, y)
    batch = 1000
    run = int(data_test.shape[0]/batch)
    if data_test.shape[0]%batch != 0:
        run += 1
    prediction_result = []
    answer_result = []
    for i in range(run):
        data = data_test[batch*i:batch*(i+1)]
        label = label_test[batch*i:batch*(i+1)]
        test_prediction, label_ = sess.run([prediction, answer], feed_dict={x: data, y: label, keep_prob: 1.0})
        prediction_result.append(test_prediction)
        answer_result.append(label_)

    prediction_result = np.concatenate(prediction_result, axis = -1)
    answer_result = np.concatenate(answer_result, axis = -1)
    confusion_matrix_ = confusion_matrix(answer_result, prediction_result)
    print("Confusion Matrix:")
    print(confusion_matrix_)
    target_names = ['Noise', 'Spike']
    report = classification_report(answer_result, prediction_result, target_names=target_names, digits=5)
    print("\nAnalysis:")
    print(report)

    if args.gather == True:
        for stage in ['train', 'val', 'test']:
            dict = np.load(os.path.join(data_path, 'npy/'+stage+'.npy'), allow_pickle=True).item()
            key = list(dict.keys())
            correct_dict = {}
            for i in range(len(key)):
                data = dict[key[i]]['signal']
                label_c = dict[key[i]]['label']
                correct_idx_file = []
                for j in range(len(data)):
                    data_ = data[j]
                    label_c_ = label_c[j]
                    label_ = np.zeros((label_c_.shape[0], 2))
                    for k in range(label_c_.shape[0]):
                        if label_c_[k, 0] == 1.0:
                            label_[k, 0] = 1
                        else:
                            label_[k, 1] = 1

                    if data_.shape[0] % batch != 0:
                        run += 1
                    prediction_result = []
                    answer_result = []
                    for l in range(run):
                        data_batch = data_[batch * l:batch * (l + 1)]
                        label_batch = label_[batch * l:batch * (l + 1)]
                        prediction_, answer_ = sess.run([prediction, answer], feed_dict={x: data_batch, y: label_batch, keep_prob: 1.0})
                        prediction_result.append(prediction_)
                        answer_result.append(answer_)

                    prediction_result = np.concatenate(prediction_result, axis=-1)
                    answer_result = np.concatenate(answer_result, axis=-1)

                    correct_idx = []
                    for l in range(prediction_result.shape[0]):
                        if prediction_result[l] == answer_result[l]:
                            correct_idx.append(l)
                    correct_idx_file.append(correct_idx)
                correct_dict[key[i]] = correct_idx_file
            np.save(os.path.join(data_path, 'npy/'+stage+'_idx.npy'), correct_dict)