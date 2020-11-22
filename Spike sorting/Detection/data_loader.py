import numpy as np
import os
from tqdm import tqdm
import scipy.io as sio
import pdb

def Load_data(number):
    print("Load: ", number)
    local_list = os.listdir('/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week/data_label/data/'+number)
    load_data = sio.loadmat('/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week/data_label/data/'+number+'/'+local_list[0])
    load_label = sio.loadmat('/media/sion/F/dataset/Prof_Kim/00.자료 공유/case2/1st week/data_label/data/'+number+'/'+local_list[1])
    SNEO = load_data['signal_data_'+number+'_SNEO_Ch2']
    SNEO = SNEO['values']
    SNEO = SNEO[0, 0]
    threshold_data = load_data['signal_data_'+number+'_SNEO_Ch6']
    threshold = threshold_data['values']
    threshold = threshold[0, 0]
    time = threshold_data['times']
    time = time[0, 0]
    label_data = load_label['signal']
    cluster = threshold_data['codes']
    cluster = cluster[0, 0]
    threshold_number = threshold.shape[0]
    window_size = threshold.shape[1]

    data = np.zeros((threshold_number, window_size, 1, 2), dtype=float)
    for i in range(threshold_number):
        data[i, ::, 0, 0] = threshold[i, ::]
        data[i, ::, 0, 1] = SNEO[int(time[i, 0] * 20000):int(time[i, 0] * 20000) + window_size, 0]

    cluster_label = np.zeros((threshold_number, max(cluster[::, 0])+1))
    label = np.zeros((threshold_number, 2), dtype=float)
    label_num = 0

    for i in range(threshold_number):
        if label_num >= label_data.shape[1]:
            label_num -= 1
        if round(data[i, 0, 0, 0], 2) == round(label_data[0, label_num], 2) and round(data[i, 1, 0, 0], 2) == \
                round(label_data[1, label_num], 2) and round(data[i, 2, 0, 0], 2) == round(label_data[2, label_num],
                                                                                           2) \
                and round(data[i, 3, 0, 0], 2) == round(label_data[3, label_num], 2) or \
                round(data[i, 0, 0, 0], 3) == round(label_data[0, label_num], 3) and round(data[i, 1, 0, 0], 3) == \
                round(label_data[1, label_num], 3) and round(data[i, 2, 0, 0], 3) == round(label_data[2, label_num],
                                                                                           3) \
                and round(data[i, 3, 0, 0], 3) == round(label_data[3, label_num], 3):
            label[i, 0] = 1
            label_num += 1
        else:
            label[i, 1] = 1

    for i in range(threshold_number):
        cluster_label[i, cluster[i, 0]] = 1

    # labeling 제대로 하는지 확인하는 용
    # error = 0
    # for i in range(label.shape[0]):
    #     if label[i, 0] == 0 and label[i, 1] == 0:
    #         error+=1
    #
    # print("error: ", error, "file: ", local_list)

    minimum_index = 0
    index = 0
    while True:
        if label[index, 0] == 1.0:
            break
        else:
            index += 1

    for i in range(window_size):
        if data[index, minimum_index, 0, 0] >= data[index, i, 0, 0]:
            minimum_index = i

    data_size = np.zeros((threshold_number, 32, 1, 2), dtype=float)
    for i in range(threshold_number):
        data_size[i, ::, ::, ::] = data[i, minimum_index - 15:minimum_index + 17, ::, ::]
    _, count = np.unique(label[:, 0], return_counts=True)

    # for i in range(data_size.shape[0]):
    #     for j in range(2):
    #         minimum = np.min(data_size[i, ::, 0, j])
    #         maximum = np.max(data_size[i, ::, 0, j])
    #         data_size[i, ::, 0, j] = (data_size[i, ::, 0, j]-minimum)/(maximum-minimum)

    return data_size, cluster_label
