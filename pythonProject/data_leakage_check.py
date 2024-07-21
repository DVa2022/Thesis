import os
import numpy as np


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def semi_supervised_data_leakage_check(index):
    train_set_folder = '/workspace/PycharmProjects/pythonProject/datasets/train' if 'workspace' in os.getcwd() else 'datasets/train'
    labels_source_dir = '/workspace/PycharmProjects/pythonProject/datasets/tracking labels/' if 'workspace' in os.getcwd() else 'datasets/tracking labels/'
    raw_frames_source_dir = '/workspace/PycharmProjects/pythonProject/datasets/raw frames/' if 'workspace' in os.getcwd() else 'datasets/raw frames/'

    curr_train_data = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/train' + index + '/labels')
    curr_valid_data = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/valid' + index + '/labels')
    curr_test_data = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/test' + index + '/labels')
    
    print('val set size: ', len(curr_valid_data))
    print('test set size: ', len(curr_test_data))
    print('train set size: ', len(curr_train_data))

    curr_valid_data = [sample[sample.find('hashed'):] for sample in curr_valid_data]
    curr_test_data = [sample[sample.find('hashed'):] for sample in curr_test_data]
    
    print('val set sample examples: ', curr_valid_data[0:3])
    print('test set sample examples: ', curr_test_data[0:3])
    print('train set sample examples: ', curr_train_data[0:3])
    
    curr_valid_data = ["_".join(sample.split('_')[:2]) for sample in curr_valid_data]
    curr_test_data = ["_".join(sample.split('_')[:2]) for sample in curr_test_data]
    curr_train_data = ["_".join(sample.split('_')[:2]) for sample in curr_train_data]

    print('val set sample examples: ', curr_valid_data[0:3])
    print('test set sample examples: ', curr_test_data[0:3])
    print('train set sample examples: ', curr_train_data[0:3])

    curr_valid_data = np.array(curr_valid_data)
    curr_test_data = np.array(curr_test_data)
    curr_train_data = np.array(curr_train_data)

    num_of_valid_videos_in_train = 0
    num_of_test_videos_in_train = 0
    for sample in curr_valid_data:
        if np.sum(curr_train_data == sample) > 0:
            num_of_valid_videos_in_train += 1
    print('Number of valid videos with at least one frame in the train set: ', num_of_valid_videos_in_train)
    for sample in curr_test_data:
        if np.sum(curr_train_data == sample) > 0:
            num_of_test_videos_in_train += 1
    print('Number of test videos with at least one frame in the train set: ', num_of_test_videos_in_train)
