import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.tensorboard import SummaryWriter
import copy
from datetime import datetime
import torchvision
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import utils
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import default_collate

from pytorch_vision_detection.utils import *
from pytorch_vision_detection.engine import evaluate


# utils.Faster_RCNN_three_annotations_per_image_correction()


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, experiment_name=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.server_flag = 'home' in os.getcwd()
        self.experiment_name = experiment_name

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']

        # open the input image
        full_image_path = os.path.join(self.root, path).replace("\\", "/")
        if self.server_flag:
            img = Image.open(full_image_path)
            if self.experiment_name == 'std':
                img_to_be_concatented = Image.open(full_image_path.replace("images/6", "images/std"))
                img = np.stack((np.array(img), np.array(img_to_be_concatented)))
            elif self.experiment_name == 'mean':
                img_to_be_concatented = Image.open(full_image_path.replace("images/6", "images/mean"))
                img = np.stack((np.array(img), np.array(img_to_be_concatented)))
            elif self.experiment_name == 'std_and_mean' or self.experiment_name == 'mean_and_std':
                img_to_be_concatented = Image.open(full_image_path.replace("images/6", "images/std"))
                img_to_be_concatented2 = Image.open(full_image_path.replace("images/6", "images/mean"))
                img = np.stack((np.array(img), np.array(img_to_be_concatented), np.array(img_to_be_concatented2)))
            elif self.experiment_name == 'three_frames':  # TODO make sure that the dimension of the channel is correct
                img = Image.open(full_image_path.replace("images/6", "images/three_frames"))
        else:  # self.experiment_name == 'base'
            img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only have one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        # a label of zero means background in COCO and/or Faster R-CNN. In out case background is non-existent objects
        my_annotation["labels"][areas == 0] = 0
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            if self.experiment_name is None or self.experiment_name == 'base' or self.experiment_name == 'three_frames':
                img = self.transforms(img)
            else:
                img = self.transforms(img).permute(1, 2, 0)

        transform = v2.ToTensor()
        tensor_img = transform(img)
        mean, std = (tensor_img.mean([1, 2]), tensor_img.std([1, 2]))
        img = v2.Normalize((mean), (std))(tensor_img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# pin_memory=True  # dataloader argument, https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

print("Imported packages.")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            targets2 = []
            for i in range(targets['boxes'].size()[0]):
                dict = {}
                dict['boxes'] = targets['boxes'][i].to(device)
                dict['labels'] = targets['labels'][i].to(device)
                dict['image_id'] = targets['image_id'][i]
                dict['area'] = targets['area'][i]
                dict['iscrowd'] = targets['iscrowd'][i]
                targets2 += [dict]
            loss_dict = model(images, targets2)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)  # doesn't do anything
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def collate_fn_mixup(batch):
    mixup = v2.MixUp(num_classes=2)
    return mixup(*default_collate(batch))


def collate_fn_cutmix(batch):
    cutmix = v2.CutMix(num_classes=2)
    return cutmix(*default_collate(batch))


def main():
    experiment_names = ['three_frames', 'std_and_mean', 'base']

    experiment_name = 'base'
    num_classes = 2
    num_epochs = 40
    max_epochs_without_improvement = 3
    data_split = [0.65, 0.20, 0.15]  # train, val, test
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    scaler = torch.cuda.amp.GradScaler if torch.cuda.is_available() else None
    batch_size = 4
    num_workers = 1 if device == torch.device('cpu') else 4
    # experiment_name = 'three_frames'
    # num_workers=os.cpu_count()
    if 'home' in os.getcwd():  # if we're on the server
        path2data = "thesisData"
        path2json = "thesisData/labels_without_multiple_labels.json"
    else:  # if we're on the PC
        path2data = "C:/Users/David/PycharmProjects/thesisData"
        path2json = "C:/Users/David/PycharmProjects/thesisData/labels_with_triple_labels.json"

    example_image_file_name = os.listdir(path2data + '/images/6')[10]
    example_image = Image.open(os.path.join(path2data + '/images/6', example_image_file_name))
    image_center_of_rotation = [0, int(example_image.width / 2)]

    # TODO be aware that some of the transformations expect the network input to be of shape [B, 1 or 3, H, W]
    # TODO consider using TrivialAugment. It's built in pytorch already

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

    # full_ds = datasets.CocoDetection(root=path2data, annFile=path2json, transform=transforms)
    full_ds_size = len(full_ds)
    generator = torch.Generator().manual_seed(42)  # Generator used for the random permutation
    images_path = path2data + '/images/6'
    indices, dataset_sizes, tagged_videos_mask = utils.find_best_data_split(images_path=images_path,
                                                                            train_share=data_split[0],
                                                                            val_share=data_split[1],
                                                                            test_share=data_split[2])
    train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)

    train_ds_size = len(train_ds)
    val_ds_size = len(val_ds)
    test_ds_size = len(test_ds)

    # changing the transformations from training transformations to inference ones
    val_ds = copy.deepcopy(val_ds)
    val_ds.dataset.transforms = transforms_inference
    test_ds = copy.deepcopy(test_ds)
    test_ds.dataset.transforms = transforms_inference

    print("Created datasets.")

    schemes = ['cutmix', 'mixup']

    for scheme in schemes:
        collate_fn = collate_fn_cutmix if scheme == 'cutmix' else collate_fn_mixup

        for i in range(0, 7):
            use_my_random_weighted_sampler = False
            if use_my_random_weighted_sampler:
                # creating the random sampler
                sex_array = np.load('/home/stu16/PycharmProjects/pythonProject/sexlist.npy')
                sex_array = sex_array[tagged_videos_mask]  # ignoring videos that aren't tagged
                train_sex_array = sex_array[indices[:train_ds_size]]  # getting only the sexes of tagged training videos
                train_sex_array[
                    train_sex_array == ''] = 'M'  # TODO the line is used only when some of the sexes aren't known
                train_sampler = utils.MyRandomWeightedSampler(data_source=train_ds, sex_mask=train_sex_array, replacement=False)
                train_loader = DataLoader(train_ds,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          sampler=train_sampler,
                                          pin_memory=True
                                          )
            else:
                train_loader = DataLoader(train_ds,
                                          shuffle=True,  # TODO remember that it was True
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          collate_fn=collate_fn
                                          )
                train_sampler = None
            val_loader = DataLoader(val_ds,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True
                                    )
            test_loader = DataLoader(test_ds,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True
                                     )

            print("Created dataloaders.")

            print('The whole dataset has {} instances'.format(full_ds_size))
            print('The training dataset has {} instances'.format(train_ds_size))
            print('The validation dataset has {} instances'.format(val_ds_size))
            print('The test dataset has {} instances'.format(test_ds_size))

            # TODO train with faster R-CNN and then with YOLOv8 and maybe with the object detection from scratch tutorial

            if experiment_name == 'base':
                experiment_name = experiment_name + '_' + scheme + '_experiment' + str(i)
            else:
                experiment_name = 'base' + '_' + scheme + '_experiment' + str(i)

            FRCNN_flag = True
            fine_tuning_flag = True  # if it's true it means that a pretrained model is used rather than training from scratch
            if FRCNN_flag:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

                # get number of input features for the classifier
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                # replace the pre-trained head with a new one
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            model.to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            # lr used to be 0.005, momentum 0.9
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

            # step size=update the learning rate every step_size epochs
            # gamma=when you update the LR, multiply it by gamma
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=3,
                                                           gamma=0.2)  # step size used to be 3 and gamma 0.1

            best_mAP = -1  # arbitrary negative value
            early_stopping_counter = 0

            only_inference = False
            if only_inference:
                file = open('/home/stu16/PycharmProjects/pythonProject/my_runs/Faster_RCNN_20230928_161212/val_loader.p',
                            'rb')
                val_loader = pickle.load(file)
                file.close()
                experiment_dir = '/home/stu16/PycharmProjects/pythonProject/my_runs/Faster_RCNN_20230928_161212'
                # best_epoch = max([int(file[6]) for file in os.listdir(experiment_dir) if 'epoch' in file])
                inference_output = utils.FASTER_RCNN_inference(val_loader=val_loader, device=device,
                                                               num_classes=num_classes,
                                                               get_IoUs=True, experiment_dir=experiment_dir,
                                                               model_path='epoch_9_AP_0.437')

            if 'my_runs' not in os.listdir(os.getcwd()):
                os.mkdir('my_runs')
            experiment_dir = '/home/stu16/PycharmProjects/pythonProject/my_runs/Faster_RCNN_' + datetime.datetime.now().strftime(
                '%Y%m%d_%H%M%S') + '_' + experiment_name
            os.mkdir(experiment_dir)
            utils.log_variables(experiment_dir, experiment_name, optimizer, train_sampler, lr_scheduler, batch_size,
                                num_classes,
                                num_epochs, device, num_workers, data_split, max_epochs_without_improvement,
                                transforms_train, train_ds_size, val_ds_size, test_ds_size, path2json)
            # saving the dataloaders
            utils.save_dls(experiment_dir, val_loader, test_loader, train_loader)

            all_mAPs = []
            all_mARs = []

            print(transforms_train)

            for epoch in tqdm(range(num_epochs)):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                writer = SummaryWriter('runs/trainer_{}'.format(timestamp))

                model.train()
                # train for one epoch, printing every 10 iterations
                train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100, scaler=None)
                # update the learning rate
                lr_scheduler.step()
                # evaluate on the test dataset
                model.eval()
                val_evaluator = evaluate(model, val_loader, device=device, mode="Validation")
                # stats is a list of the average precisions and average recalls. The first entry is the mAP over IoU=0.5:0.95
                mAP = val_evaluator.coco_eval['bbox'].stats[0]
                mAR = val_evaluator.coco_eval['bbox'].stats[6]  # mean average recall over IoU=0.5:0.95

                # Track the best performance, and save the model's state
                if mAP > best_mAP:
                    all_mAPs += [mAP]
                    all_mARs += [mAR]
                    early_stopping_counter = 0
                    best_mAP = mAP
                    model_path = 'epoch_{}_AP_{}'.format(epoch, str(mAP)[:5])
                    # torch.save(model.state_dict(), experiment_dir + '/' + model_path)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter == max_epochs_without_improvement:
                        print(
                            str(max_epochs_without_improvement) + " epochs without an improvement in the mAP. Stopping the training")
                        np.save(experiment_dir + "/all_mAPs", np.array(all_mAPs))
                        np.save(experiment_dir + "/all_mARs", np.array(all_mARs))
                        np.savetxt(experiment_dir + '/all_mAPs.csv', np.array(all_mAPs), delimiter=',')
                        plt.plot(all_mAPs, label='mAPs')
                        plt.plot(all_mARs, label='mARs')
                        plt.legend(loc="upper left")
                        plt.xlabel('Epoch number')
                        plt.savefig(experiment_dir + '/performance_per_epoch.png')
                        plt.clf()  # clearing the plot for the next plot
                        break


    inference_output = utils.FASTER_RCNN_inference(val_loader=val_loader, device=device, num_classes=num_classes,
                                                   experiment_dir=experiment_dir,
                                                   model_path=model_path + '_' + experiment_name, get_IoUs=True)

    test_evaluator = evaluate(model, test_loader, device=device, mode="Test")
    test_mAP = test_evaluator.coco_eval['bbox'].stats[0]


if __name__ == '__main__':
    main()
