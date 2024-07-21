from ultralytics import YOLO
import torch
from torchvision.transforms import v2
from CNN import myOwnDataset
import utils
import numpy as np
import os
import shutil
import random
import datetime
import pickle
import matplotlib.pyplot as plt

# model_sizes = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']


def delete_prev_models():
    """
    This function deletes the weight files of all previously trained YOLO models.
    It's used to save disk space on the server.

    :return: None.
    """
    return
    experiment_dirs = [dir for dir in os.listdir('runs/detect') if '_take_' in dir]
    for dir in experiment_dirs:
        if 'weights' in os.listdir(os.path.join('runs/detect', dir)):
            shutil.rmtree('runs/detect/' + dir + '/weights')


def save_datasets(run_dir):
    """
    This function saves the dataset split found in the current test, valid and train folders.
    The split is saved as three numpy files, each containing the names of the image files of the subset.

    :param run_dir: The folder of the current run in the experiment.
    :return: None.
    """

    base_dir = 'datasets/'
    test_set = np.array(os.listdir(base_dir + 'test/labels'))
    valid_set = np.array(os.listdir(base_dir + 'valid/labels'))
    train_set = np.array(os.listdir(base_dir + 'train/labels'))

    saving_dir = 'runs/detect/' + run_dir + '/'

    np.save(saving_dir + 'test_set', test_set)
    np.save(saving_dir + 'valid_set', valid_set)
    np.save(saving_dir + 'train_set', train_set)


def yolo_to_pascal_voc(x_center, y_center, w, h):
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x1 + w
    y2 = y1 + h

    return np.array([x1, y1, x2, y2])


def get_val_results(model, saving_dir, valid_or_test='valid'):
    """
    This function does inference on the images in the validation images folder and calculates the IoU for each image
    using the ground truth labels and the predictions.
    :param valid_or_test: A string which indicates whether to do inference on the validation or test set.
    :param model: A YOLO model that is used to do inference on the validation set images.
    :param saving_dir: The folder to which the IoU distribution is saved. Generally, it's supposed to be the folder of
    the current run.
    :return: None
    """
    
    if 'series1' in saving_dir:
        dir_num = ''
    if 'series2' in saving_dir:
        dir_num = '2'
    if 'series3' in saving_dir:
        dir_num = '3'
    
    on_dgx = 'workspace' in os.getcwd()
    datasets_path = '/workspace/PycharmProjects/pythonProject/datasets/' if on_dgx else 'datasets/'
    models_path = '/workspace/runs/detect/' if on_dgx else 'runs/detect/'

    # the paths of all images in the validation images folder
    val_images = os.listdir(datasets_path + valid_or_test + dir_num + '/images')
    val_images_full_path = [datasets_path + valid_or_test + dir_num + '/images/' + name for name in
                            val_images]
    # val_results = [model(val_image)[0] for val_image in val_images_full_path]
    # the paths of all labels in the validation labels folder
    val_labels_full_path = [path.replace('jpg', 'txt').replace('images', 'labels') for path in val_images_full_path]

    IoUs = []
    for val_label_path, val_image_path in zip(val_labels_full_path, val_images_full_path):
        # getting the predicted bounding box
        boxA = model(val_image_path)[0].boxes.xywhn.cpu().numpy()
        if boxA.size == 0:
            IoUs += [0]
            continue
        else:
            boxA = boxA[0]
        boxA = yolo_to_pascal_voc(boxA[0], boxA[1], boxA[2], boxA[3])

        # getting the ground truth bounding box
        fl = open(val_label_path)
        data = fl.readlines()
        fl.close()
        for dt in data:
            _, x, y, w, h = map(float, dt.split(' '))  # Split string to float
        boxB = np.array([x, y, w, h])
        boxB = yolo_to_pascal_voc(x, y, w, h)

        # calculating the IoU
        IoUs += [utils.boxes_shared_area(boxA, boxB, IoU=True)]

    mean_IoU = np.array(IoUs).mean()

    print('Mean ' + valid_or_test + ' IoU is ', mean_IoU)
    hist = plt.hist(IoUs, bins=20)
    plt.xlabel("IoU")
    plt.ylabel("Number of videos")
    plt.title("IoU Distribution. Mean IoU: " + str(mean_IoU)[:5])
    plt.savefig(models_path + saving_dir + '/' + valid_or_test + '_all_video_IoUs.png')
    plt.clf()


def main():

    train_sets = []
    for i in range(17, 22):  # these numbers correspond to the previous models that have weights and train_set files
        train_sets += [np.load(
            'runs/detect/fourth_take_no_dfl_extra_large_model' + str(i) + '/train_set.npy')]

    sample_dict = {}
    current_sample_list = []
    # creating a dictionary that maps each sample name to a list of models that were trained on the sample
    for sample in os.listdir('datasets/labels'):
        for train_set_num, train_set in enumerate(train_sets):
            if sample in train_set:
                current_sample_list += [str(train_set_num + 17)]
        sample_dict[sample] = current_sample_list
        current_sample_list = []
    prev_tracked_videos = os.listdir('datasets/tracking labels')
    for annotation_file_name, suitable_models_list in zip(sample_dict.keys(), sample_dict.values()):
        split_file_name = annotation_file_name.split('_')
        file_name_ending = split_file_name[-1]
        annotated_frame = file_name_ending[:file_name_ending.find('.')]  # number of the annotated frame in the video
        hashed_name = split_file_name[-3] + '_' + split_file_name[-2]  # ='hashed_...'
        if hashed_name in prev_tracked_videos:  # there's no need to do tracking on videos that were already tracked
            continue
        model_number = suitable_models_list[0]  # arbitrarily choose the first model in the list
        model_dir = 'fourth_take_no_dfl_extra_large_model' + model_number
        utils.one_video_object_tracking(annotated_frame, model_dir=model_dir, dicom_name=hashed_name,
                                  not_annotated_numpy_frames_folder=None, save_images=True,
                                  save_video=False, save_annotation=True)

    experiment_name = 'initial'
    num_epochs = 40
    max_epochs_without_improvement = 7
    data_split = [0.65, 0.20, 0.15]  # train, val, test
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 4
    num_workers = 1 if device == torch.device('cpu') else 4
    path2data = "thesisData"
    path2json = "thesisData/labels_without_multiple_labels.json"

    prev_indices = torch.zeros((2, 50))

    transforms_train = v2.Compose([
        v2.ToTensor(),
        # TODO it normalizes the values. Thus, it shouldn't be used for image masks. See this for image masks: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html#torchvision.transforms.v2.ToTensor
        v2.RandomEqualize(p=0.35),  # TODO consider applying p=1
    ])
    transforms_inference = v2.Compose([
        v2.ToTensor(),
    ])

    full_ds = myOwnDataset(root=path2data,
                           annotation=path2json,
                           experiment_name=experiment_name,
                           transforms=transforms_train
                           )

    # generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))  # Generator used for the random permutation
    images_path = path2data + '/images/6'
    indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                            train_share=data_split[0],
                                                                            val_share=data_split[1],
                                                                            test_share=data_split[2],
                                                                            prev_indices=prev_indices)
    train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split)
    utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)

    # cache, imgsz, deterministic, nbs (nominal batch size)
    # device = [0, 1]

    # delete_prev_models()

    print('no_dfl experiment')
    mAPs10 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_no_dfl',
                              optimizer='SGD', lr0=0.005, lrf=0.05, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0, dfl=0)
        mAPs10 += [results.box.ap[0]]
        torch.Generator().manual_seed(999)
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs10) < 0.82:
            break


    print('no_cls experiment')
    mAPs11 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_no_cls',
                              optimizer='SGD', lr0=0.005, lrf=0.05, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0, cls=0)
        mAPs11 += [results.box.ap[0]]
        torch.Generator().manual_seed(999)
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs11) < 0.82:
            break


    # TODO watch these:
    # seed = int(torch.empty((), dtype=torch.int64).random_().item())
    # generator = torch.Generator()
    # generator.manual_seed(seed)


    print('no_dfl experiment, long_training')
    mAPs10 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=120, imgsz=640,
                              patience=20,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_no_dfl_long_training_smaller_lr',
                              optimizer='SGD', lr0=0.005, lrf=0.000125, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, dfl=0)
        mAPs10 += [results.box.ap[0]]
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        # rand_seed = torch.randint(low=1, high=10000, size=(1,))[0].item()
        print("Random seed: ", seed)
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs10) < 0.82:
            break


    # delete_prev_models()


    mAPs = []
    mAPs2 = []
    mAPs3 = []
    print('base experiment')
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_base_experiment',
                              optimizer='SGD', lr0=0.005, lrf=0.05, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                              scale=0, fliplr=0, mosaic=0)
        mAPs += [results.box.ap[0]]
        rand_seed = torch.randint(low=1, high=10000, size=(1,))[0].item()
        print("Random seed: ", rand_seed)
        generator = torch.Generator().manual_seed(rand_seed)
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
    mean_mAPs = np.array(mAPs).mean()

    # delete_prev_models()


    print('base experiment, long training')
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=120, imgsz=640,
                              patience=20,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_base_experiment_long_training_smaller_lr',
                              optimizer='SGD', lr0=0.005, lrf=0.000125, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                              scale=0, fliplr=0, mosaic=0)
        mAPs += [results.box.ap[0]]
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        # rand_seed = torch.randint(low=1, high=10000, size=(1,))[0].item()
        print("Random seed: ", seed)
        # rand_seed = torch.randint(low=1, high=10000, size=(1,))[0].item()

        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
    mean_mAPs = np.array(mAPs).mean()


    # print('AdamWOptimizer experiment')
    # for i in range(9):
    #     model = YOLO("yolov8m.pt")
    #     # lr used to be 0.005, momentum 0.9
    #     # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    #
    #     # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
    #     # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]
    #
    #     results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
    #                           patience=max_epochs_without_improvement,
    #                           batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_AdamWOptimizer',
    #                           lr0=0.005, lrf=0.05, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
    #                           scale=0, fliplr=0, mosaic=0)
    #     mAPs2 += [results.box.ap[0]]
    #     generator = torch.Generator().manual_seed(random.randint(1, 1000))
    #     indices, dataset_sizes, tagged_videos_mask = utils.find_best_data_split(images_path=images_path,
    #                                                                             train_share=data_split[0],
    #                                                                             val_share=data_split[1],
    #                                                                             test_share=data_split[2])
    #     train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
    #     utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
    #     file_number = '' if i == 0 else str(i + 1)
    #     weight_files = os.listdir('runs/detect/second_take_AdamWOptimizer' + file_number + '/weights')
    #     for file in weight_files:
    #         removed = os.remove('runs/detect/second_take_AdamWOptimizer' + file_number + '/weights/' + file)
    #     if i == 1 and sum(mAPs2) < 0.82:
    #         break
    # mean_mAPs2 = np.array(mAPs2).mean()

    # delete_prev_models()

    print('default_LR experiment')
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_default_LR',
                              optimizer='SGD', hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                              scale=0, fliplr=0, mosaic=0)
        mAPs3 += [results.box.ap[0]]
        generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs3) < 0.82:
            break
    mean_mAPs3 = np.array(mAPs3).mean()

    # delete_prev_models()

    print('mosaicing experiment')
    mAPs4 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_mosaicing',
                              lr0=0.005, lrf=0.05,
                              optimizer='SGD', hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                              scale=0, fliplr=0)
        mAPs4 += [results.box.ap[0]]
        generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs4) < 0.82:
            break
    mean_mAPs4 = np.array(mAPs4).mean()

    # delete_prev_models()

    print('horizontal_flip experiment')
    mAPs5 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_horizontal_flip',
                              optimizer='SGD', lr0=0.005, lrf=0.05, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                              scale=0, mosaic=0)
        mAPs5 += [results.box.ap[0]]
        generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs5) < 0.82:
            break
    mean_mAPs5 = np.array(mAPs5).mean()

    # delete_prev_models()

    print('mixup experiment')
    mAPs6 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_mixup',
                              optimizer='SGD', lr0=0.005, lrf=0.05, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0.5, fliplr=0)
        mAPs6 += [results.box.ap[0]]
        generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs6) < 0.82:
            break
    mean_mAPs6 = np.array(mAPs6).mean()

    # delete_prev_models()

    print('hsv_augmentation experiment')
    mAPs7 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=num_epochs, imgsz=640,
                              patience=max_epochs_without_improvement,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_hsv_augmentation',
                              optimizer='SGD', lr0=0.005, lrf=0.05, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0)
        mAPs7 += [results.box.ap[0]]
        torch.Generator().manual_seed(999)
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs7) < 0.82:
            break
    mean_mAPs7 = np.array(mAPs7).mean()

    # delete_prev_models()


    print('cosine_lr experiment')
    mAPs8 = []
    for i in range(0):
        model = YOLO("yolov8m.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=80, imgsz=640,
                              patience=15, cos_lr=True,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_cosine_lr',
                              optimizer='SGD', lr0=0.005, lrf=0.05, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0)
        mAPs8 += [results.box.ap[0]]
        generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
        if i == 1 and sum(mAPs8) < 0.82:
            break
    mean_mAPs8 = np.array(mAPs8).mean()

    mAPs12 = []
    print('no_dfl experiment, large model')
    for i in range(0):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        # rand_seed = torch.randint(low=1, high=10000, size=(1,))[0].item()
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)

        if i == 6:
            delete_prev_models()

        model = YOLO("yolov8l.pt")

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        results = model.train(data='data.yaml', epochs=80, imgsz=640,
                              patience=10,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_no_dfl_large_model',
                              optimizer='SGD', lr0=0.003, lrf=0.03, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, dfl=0)
        mAPs12 += [results.box.ap[0]]

    # delete_prev_models()

    mAPs13 = []
    print('no_dfl experiment, extra large model')
    for i in range(0):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
                                                                               train_share=data_split[0],
                                                                               val_share=data_split[1],
                                                                               test_share=data_split[2],
                                                                               prev_indices=prev_indices)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator)
        utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)

        model = YOLO("yolov8x.pt")

        # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
        # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]

        prev_dirs = [dir for dir in os.listdir('runs/detect')]
        results = model.train(data='data.yaml', epochs=90, imgsz=640,
                              patience=14,
                              batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_no_dfl_extra_large_model',
                              optimizer='SGD', lr0=0.0016, lrf=0.025, degrees=0, translate=0,
                              scale=0, mosaic=0, mixup=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, dfl=0)
        new_dirs = [dir for dir in os.listdir('runs/detect')]
        newest_dir = [dir for dir in new_dirs if not dir in prev_dirs][0]
        save_datasets(run_dir=newest_dir)
        model = YOLO('runs/detect/' + newest_dir + '/weights/best.pt')
        get_val_results(model, saving_dir=newest_dir, valid_or_test='valid')
        get_val_results(model, saving_dir=newest_dir, valid_or_test='test')

        # # delete all previous weight files if the last model is better than all previous ones
        # if i >= 2 and results.box.ap[0] > max(mAPs13):
        #     for dir in prev_dirs:
        #         if 'weights' in os.listdir(os.path.join('runs/detect', dir)):
        #             shutil.rmtree('runs/detect/' + dir + '/weights')

        mAPs13 += [results.box.ap[0]]


    # tracking
    # import cv2
    #
    # video_path = '.mp4'
    # cap = cv2.VideoCapture(video_path)
    #
    # while ret:
    #     ret, frame = cap.read()
    #     if ret:
    #         tracking_res = model.track(source="", show=True)
    #         frame_ = tracking_res[0].plot()
    #         cv2.imshow('frame', frame_)
    #         if cv2.waitKey(25) and 0xFF == ord('q'):
    #             break

    print('dog')
    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category

    # results = model.predict("cat_dog.jpg")


if __name__ == '__main__':
    # file_names = ['hashed_1247406082', 'hashed_1239853307', 'hashed_1251305651', 'hashed_1250314470', 'hashed_1240974238', 'hashed_1238264340', 'hashed_1232916126', 'hashed_1191170364', 'hashed_1243190299', 'hashed_1223085847', 'hashed_1245583517', 'hashed_1191356738', 'hashed_1245044876', 'hashed_1191371648', 'hashed_1256396883', 'hashed_1238611676', 'hashed_1234710597', 'hashed_1176050251', 'hashed_1194809127', 'hashed_1218208143', 'hashed_1191138030', 'hashed_1251305681', 'hashed_1245740461', 'hashed_1250593131', 'hashed_1246274399', 'hashed_1238104877', 'hashed_1194414681', 'hashed_1224186406', 'hashed_1227596406', 'hashed_1217790808']
    # annotated_frames = utils.get_tagged_frame_index(file_names)
    # for file_name, annotated_frame in zip(file_names, annotated_frames):
    #     utils.one_video_object_tracking(annotated_frame=annotated_frame,
    #                                     model_dir='fourth_take_no_dfl_extra_large_model14',
    #                                     dicom_name=file_name,
    #                                     not_annotated_numpy_frames_folder=None, save_images=True,
    #                                     save_video=True, save_annotation=True, last_frame=None)
    main()