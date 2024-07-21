from ultralytics import YOLO
# from my_yolo_model import YOLO
import torch
from torchvision.transforms import v2
from CNN import myOwnDataset
from utils import *
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import datetime
import shutil
import numpy as np
from data_leakage_check import semi_supervised_data_leakage_check

# from yolo_experiment import delete_prev_models, save_datasets, yolo_to_pascal_voc, get_val_results


def main():

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    print(torch.cuda.device_count())
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    
    torch.distributed.init_process_group(backend='nccl')
    
    current_ds = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/train/labels')
    original_ds = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/labels')
    original_ds = [file.split('_')[3] + '_' + file.split('_')[4] + '_' + file.split('_')[5] for file in original_ds]
    print('example of original_ds values: ', original_ds[:3])
    for file in current_ds:
        if 'hashed' in file and file not in original_ds:
            os.remove('/workspace/PycharmProjects/pythonProject/datasets/train/labels/' + file)
            os.remove('/workspace/PycharmProjects/pythonProject/datasets/train/images/' + file.replace('txt', 'jpg'))
    
    max_num_of_frames_from_each_vid_options = [52]
    for max_num_of_frames_from_each_vid in max_num_of_frames_from_each_vid_options:
        
        experiment_name = 'semi-supervised_XL_model_training-series1_' + str(max_num_of_frames_from_each_vid) + '-fold_increase_with_unannotated_data_'
        num_epochs = 55
        max_epochs_without_improvement = 11
        # ... + int(math.sqrt(max_num_of_frames_from_each_vid))
        data_split = [0.65, 0.20, 0.15]  # train, val, test
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        batch_size = 16
        num_workers = 1 if device == torch.device('cpu') else 24
        path2data = "thesisData"
        path2json = "thesisData/labels_without_multiple_labels.json"
    
        # prev_indices = torch.zeros((2, 50))
    
        transforms_train = v2.Compose([
            v2.ToTensor(),
            # TODO it normalizes the values. Thus, it shouldn't be used for image masks. See this for image masks: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html#torchvision.transforms.v2.ToTensor
            v2.RandomEqualize(p=0.35),  # TODO consider applying p=1
        ])
        # transforms_inference = v2.Compose([
        #     v2.ToTensor(),
        # ])
        #
        # full_ds = myOwnDataset(root=path2data,
        #                        annotation=path2json,
        #                        experiment_name=experiment_name,
        #                        transforms=transforms_train
        #                        )
    
        # generator = torch.Generator().manual_seed(int(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[-3:]))  # Generator used for the random permutation
        # images_path = path2data + '/images/6'
        # indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
        #                                                                         train_share=data_split[0],
        #                                                                         val_share=data_split[1],
        #                                                                         test_share=data_split[2],
        #                                                                         prev_indices=prev_indices)
        # train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split)
        # utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
    
        # cache, imgsz, deterministic, nbs (nominal batch size)
        # device = [0, 1]
    
        # delete_prev_models()
    
        training_order_dict = create_all_vids_train_order(use_unannotated_scans=True)
        semisupervised_training_set_prep(training_order_dict, max_num_of_frames_from_each_vid)
        semi_supervised_data_leakage_check('')
            
        num_of_samples = len(os.listdir('/workspace/PycharmProjects/pythonProject/datasets/train/labels'))
        experiment_name = experiment_name + str(num_of_samples)
        
        model = YOLO("yolov8x.pt")
        # lr used to be 0.005, momentum 0.9
        # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
        results = model.train(data='/workspace/PycharmProjects/pythonProject/datasets/data.yaml', epochs=num_epochs, imgsz=640, patience=max_epochs_without_improvement, batch=batch_size, cache=True, device=device, workers=num_workers, name=experiment_name, optimizer='SGD', lr0=0.005, lrf=0.05, degrees=0, translate=0, scale=0, mosaic=0, mixup=0, fliplr=0, dfl=0, warmup_epochs=0)
        
            # torch.Generator().manual_seed(94499)
            # indices, dataset_sizes, _, prev_indices = utils.find_best_unique_split(images_path=images_path,
            #                                                                        train_share=data_split[0],
            #                                                                        val_share=data_split[1],
            #                                                                        test_share=data_split[2],
            #                                                                        prev_indices=prev_indices)
            # train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split)
            # utils.prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
    dist.destroy_process_group()

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