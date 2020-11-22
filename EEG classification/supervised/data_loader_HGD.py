import tensorflow as tf
import numpy as np
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import pdb
import resampy

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, axis = -1, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.filtfilt(b, a, data, axis=axis)
    # y = sig.lfilter(b, a, data, axis = axis)
    return y

def minmax01_norm(x):
    ## dimension is (number, sampling, channel, freq)
    min = np.expand_dims(np.min(x, axis = 1), axis = 1)
    max = np.expand_dims(np.max(x, axis = 1), axis = 1)
    norm = (x-min)/(max-min)

    return norm

def data_load(path, subject, mode, model):
    with tf.variable_scope('Load_Data'):
        path = path
        data_list = np.sort(os.listdir(path+'/data/HGD_3ch'))
        label_list = np.sort(os.listdir(path+'/label/HGD_3ch'))
        subject_num = subject
        data_dict = {}
        label_dict = {}
        for i in range(len(data_list)):
            file_name = data_list[i].split('.')[0]
            if mode == 'train':
                if file_name == str(subject_num):
                    print("Load", data_list[i])
                    data = np.float32(np.load(path+'/data/HGD_3ch/'+data_list[i]))
                    label = np.float32(np.load(path + '/label/HGD_3ch/' + data_list[i]))

                    count = 0
                    for j in range(data.shape[0]):
                        if int(np.argmax(label[j])) < 2:
                            count += 1

                    data_two = np.float32(np.zeros((count, data.shape[1], data.shape[2])))
                    label_two = np.float32(np.zeros((count, 2)))

                    count = 0
                    for j in range(data.shape[0]):
                        if int(np.argmax(label[j])) < 2:
                            data_two[count] = data[j]
                            # print(abs(int(np.argmax(label[j]))-1))
                            label_two[count, abs(int(np.argmax(label[j]))-1)] = 1 #left : 1, right: 2
                            count += 1

                    data_two = butter_bandpass_filter(data_two, 0.5, 100, 250, -1, 4)
                    data_two = data_two[:, :, :2000]
                    data = data_two[:, :, ::2]
                    label = label_two

                    data = np.expand_dims(np.swapaxes(data, 1, 2), axis = 3)
                    if model == "HS_CNN":
                        data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                        data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                        data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                        data = np.concatenate((data_47, data_813, data_1332), axis=-1)
                    for j in range(data.shape[0]):
                        mean = np.mean(data[j])
                        std = np.std(data[j])
                        data[j] = (data[j] - mean) / std
                        #max = np.max(data[j])
                        #min = np.min(data[j])
                        #data[j] = (data[j] - min) / (max - min)

                    data_dict[data_list[i]] = data
                    label_dict[data_list[i]] = label
                    #pdb.set_trace()

            if mode == 'test':
                if file_name == str(subject_num)+'_test':
                    print("Load", data_list[i])
                    data = np.float32(np.load(path + '/data/HGD_3ch/' + data_list[i]))
                    label = np.float32(np.load(path + '/label/HGD_3ch/' + data_list[i]))

                    count = 0
                    for j in range(data.shape[0]):
                        if int(np.argmax(label[j])) < 2:
                            count += 1

                    data_two = np.float32(np.zeros((count, data.shape[1], data.shape[2])))
                    label_two = np.float32(np.zeros((count, 2)))

                    count = 0
                    for j in range(data.shape[0]):
                        if int(np.argmax(label[j])) < 2:
                            data_two[count] = data[j]
                            label_two[count, abs(int(np.argmax(label[j]))-1)] = 1 #left : 1, right: 2
                            count += 1
                    data_two = butter_bandpass_filter(data_two, 0.5, 100, 500, -1, 4)
                    data_two = data_two[:, :, :2000]
                    data = data_two[:, :, ::2]
                    label = label_two

                    data = np.expand_dims(np.swapaxes(data, 1, 2), axis=3)
                    if model == "HS_CNN":
                        data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                        data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                        data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                        data = np.concatenate((data_47, data_813, data_1332), axis=-1)

                    for j in range(data.shape[0]):
                        mean = np.mean(data[j])
                        std = np.std(data[j])
                        data[j] = (data[j] - mean)/std
                        #max = np.max(data[j])
                        #min = np.min(data[j])
                        #data[j] = (data[j]-min)/(max-min)

                    data_dict[data_list[i]] = data
                    label_dict[data_list[i]] = label
                    del data
                    del label

        return data_dict, label_dict

# path = 'F:/dataset\\EEG\\numpy'
# subject_num = 1
# data, label, data_raw = data_load(path, subject_num)
# print(data.shape)
# print(data_raw.shape)
# plt.plot(data_raw[0, :, 0, 0], 'k')
# plt.axis('off')
# plt.show()
# plt.clf()
# plt.plot(data_raw[0, :, 1, 0], 'k')
# plt.axis('off')
# plt.show()
# plt.clf()
# plt.plot(data_raw[0, :, 2, 0], 'k')
# plt.axis('off')
# plt.show()
# plt.clf()
# plt.plot(data[0, :, 0, 0], 'k')
# plt.axis('off')
# plt.show()
# plt.clf()
# plt.plot(data[0, :, 0, 1], 'k')
# plt.axis('off')
# plt.show()
# plt.clf()
# plt.plot(data[0, :, 0, 2], 'k')
# plt.axis('off')
# plt.show()
# plt.clf()
