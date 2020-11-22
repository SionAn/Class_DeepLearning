import tensorflow as tf
import numpy as np
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import pdb

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, axis = -1, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.filtfilt(b, a, data, axis = axis)
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
        data_list = os.listdir(path+'/data/BCI4_2b')
        label_list = os.listdir(path+'/label/BCI4_2b')
        data_list.sort()
        label_list.sort()
        subject_num = subject
        data_dict = {}
        label_dict = {}
        for i in range(len(data_list)):
            if mode == 'train':
                if data_list[i][2] == str(subject_num) and data_list[i][5] == 'T':
                    print("Load: ", data_list[i])
                    data = np.load(path+'/data/BCI4_2b/'+data_list[i])
                    data = np.expand_dims(np.swapaxes(data, 1, 2), axis = 3)
                    data = data[::, ::, 0:3]
                    # pdb.set_trace()
                    # label = np.load(path + '/label/BCI4_2b/' + label_list[i])
                    # if not model == 'HS_CNN':
                    #     data = butter_bandpass_filter(data, 3, 40, 250, 1, 4)  # mne.filter.filter_data(data, 250, 3, 40, method='iir', phase='zero')

                    if model == 'HS_CNN':
                        data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                        data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                        data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                        data = np.concatenate((data_47, data_813, data_1332), axis=-1)
                    for j in range(data.shape[0]):
                        mean = np.mean(data[j])
                        std = np.std(data[j])
                        data[j] = (data[j] - mean) / std
                    #     max = np.max(data[j])
                    #     min = np.min(data[j])
                    #     data[j] = (data[j] - min) / (max - min)

                    data_dict[data_list[i]] = data
                    label_dict[data_list[i]] = np.load(path + '/label/BCI4_2b/' + label_list[i])

            if mode == 'test':
                if data_list[i][2] == str(subject_num) and data_list[i][5] == 'E':
                    print("Load: ", data_list[i])
                    data = np.load(path + '/data/BCI4_2b/' + data_list[i])
                    data = np.expand_dims(np.swapaxes(data, 1, 2), axis=3)
                    data = data[::, ::, 0:3]
                    # if not model == 'HS_CNN':
                    #     data = butter_bandpass_filter(data, 3, 40, 250, 1, 4)#mne.filter.filter_data(data, 250, 3, 40, method='iir', phase='zero')

                    if model == 'HS_CNN':
                        data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                        data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                        data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                        data = np.concatenate((data_47, data_813, data_1332), axis=-1)

                    for j in range(data.shape[0]):
                        mean = np.mean(data[j])
                        std = np.std(data[j])
                        data[j] = (data[j] - mean) / std
                    #     max = np.max(data[j])
                    #     min = np.min(data[j])
                    #     data[j] = (data[j] - min) / (max - min)
                    data_dict[data_list[i]] = data
                    label_dict[data_list[i]] = np.load(path + '/label/BCI4_2b/' + label_list[i])

        return data_dict, label_dict

