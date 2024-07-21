import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.tensorboard import SummaryWriter
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
# from ray import tune

from pytorch_vision_detection.utils import *
from pytorch_vision_detection.engine import evaluate
from pytorch_vision_detection.coco_eval import CocoEvaluator
from pytorch_vision_detection.coco_utils import get_coco_api_from_dataset

# PyTorch TensorBoard support

# utils.Faster_RCNN_three_annotations_per_image_correction()


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.server_flag = 'home' in os.getcwd()

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
        if self.server_flag:
            img = Image.open(os.path.join(self.root, path).replace("\\", "/"))
        else:
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
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# pin_memory=True  # dataloader argument, https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

print("Imported packages.")


# def test_epoch(model, dl, FRCNN_flag=True):
#     total_loss = 0.
#
#     with torch.no_grad():
#         for i, data in enumerate(dl):
#             # Every data instance is an input + label pair
#             inputs, labels = data
#
#             # Make predictions for this batch
#             outputs = model(inputs)
#
#     return total_loss


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


def collate_fn(batch):
    return torch.stack(batch, dim=0)


def main():
    num_classes = 2
    num_epochs = 10
    data_split = [0.7, 0.3, 0.0]  # train, val, test
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    scaler = torch.cuda.amp.GradScaler if torch.cuda.is_available() else None
    batch_size = 4
    num_workers = 1 if device == torch.device('cpu') else 4
    # num_workers=os.cpu_count()
    if 'home' in os.getcwd():  # if we're on the server
        path2data = "thesisData"
        path2json = "thesisData/10_samples.json"
    else:  # if we're on the PC
        path2data = "C:/Users/David/PycharmProjects/thesisData"
        path2json = "C:/Users/David/PycharmProjects/thesisData/labels_with_triple_labels.json"

    example_image_file_name = os.listdir(path2data + '/images/6')[8]
    example_image = Image.open(os.path.join(path2data + '/images/6', example_image_file_name))
    image_center_of_rotation = [0, int(example_image.width / 2)]

    # TODO check the bounding box augmentation
    # TODO be aware that some of the transformations expect the network input to be of shape [B, 1 or 3, H, W]
    # TODO consider using TrivialAugment. It's built in pytorch already
    transforms_train = v2.Compose([
        v2.ToTensor(),
        # TODO it normalizes the values. Thus, it shouldn't be used for image masks. See this for image masks: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html#torchvision.transforms.v2.ToTensor
        # "Convert a PIL Image or ndarray (H x W x C) to tensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image... or if the numpy.ndarray has dtype = np.uint8"
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAutocontrast(p=0.5),
        v2.ColorJitter(),  # TODO maybe shouldn't use it
        v2.RandomRotation(degrees=(-20, 20), interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                          expand=False, center=image_center_of_rotation),
        v2.RandomEqualize(p=0.5),  # TODO consider applying p=1
        # v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485], std=[0.229]),
    ])

    transforms_inference = v2.Compose([
        v2.ToTensor()
    ])

    full_ds = myOwnDataset(root=path2data,
                           annotation=path2json,
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

    # creating the random sampler
    sex_array = np.load('sexlist.npy')
    sex_array = sex_array[tagged_videos_mask]  # ignoring videos that aren't tagged
    train_sex_array = sex_array[indices[:train_ds_size]]  # getting only the sexes of tagged training videos
    train_sex_array[train_sex_array == ''] = 'M'  # TODO the line is used only when some of the sexes aren't known
    train_sampler = utils.MyRandomWeightedSampler(data_source=train_ds, sex_mask=train_sex_array, replacement=False)

    # changing the transformations from training transformations to inference ones
    val_ds.dataset.transforms = transforms_inference
    test_ds.dataset.transforms = transforms_inference

    print("Created datasets.")

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              )
    # train_loader = DataLoader(train_ds,
    #                           batch_size=batch_size,
    #                           num_workers=num_workers,
    #                           sampler=train_sampler
    #                           )
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers
                            )
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers
                             )

    print("Created dataloaders.")

    FRCNN_flag = True
    if FRCNN_flag:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    print('The whole dataset has {} instances'.format(full_ds_size))
    print('The training dataset has {} instances'.format(train_ds_size))
    print('The validation dataset has {} instances'.format(val_ds_size))
    print('The test dataset has {} instances'.format(test_ds_size))

    # TODO train with faster R-CNN and then with YOLOv8 and maybe with the object detection from scratch tutorial

    best_mAP = -1  # arbitrary negative value
    loss_fn = torch.nn.CrossEntropyLoss()  # should be changed
    early_stopping_counter = 0

    for epoch in tqdm(range(num_epochs)):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/trainer_{}'.format(timestamp))

        model.train()



        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10, scaler=None)
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
            early_stopping_counter = 0
            best_mAP = mAP
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter == 10:
                print("10 epochs without an improvement in the mAP. Stopping the training")
                break

        print("\n")

    test_evaluator = evaluate(model, test_loader, device=device, mode="Test")
    test_mAP = test_evaluator.coco_eval['bbox'].stats[0]
    print(test_mAP)


if __name__ == '__main__':
    main()

#
# def forward_epoch(model, dl, loss_function, optimizer, total_loss=0,
#                   to_train=False, desc=None, device=torch.device('cpu'), weighting=False, bce2=False):
#
#     # total loss is over the entire epoch
#     # y_trues is by patient for the entire epoch; can get last batch with [-batch_size]
#     # y_preds is by patient for the entire epoch
#     #
#     with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
#         model = model.double().to(device)  # solving runtime memory issue
#
#         y_trues = torch.empty(0).type(torch.int).to(device)
#         y_preds = torch.empty(0).type(torch.int).to(device)
#         for i_batch, (X, y) in enumerate(dl):
#             # print('sickyall',X.dtype)
#             X = X.to(device)
#             X = X.type(torch.double)
#             # print('wackyall',X.dtype)
#             y = y.to(device)
#
#             # Forward:
#             y_pred = model(X)
#
#             # Loss:
#             y_true = y.type(torch.double)
#             if weighting: #loss_function.reduction == 'none':
#                 loss = bce2(y_pred, y_true)  # straight mean
#                 lossbatch = loss_function(y_pred, y_true)  # loss of one batch per batch element
#                 y_predn = y_pred.cpu().detach().numpy()
#                 y_truen = y_true.cpu().detach().numpy()
#                 lossn = lossbatch.cpu().detach().numpy()
#                 print()
#                 print(y_predn)
#                 print(y_truen) # TODO get weighting
#                 mask0 = ((y_predn != y_truen) + (y_truen == 0)) == 2
#                 mask1 = ((y_predn != y_truen) + (y_truen == 1)) == 2
#                 lossn[mask0] = weighting[0]*lossn[mask0]
#                 lossn[mask1] = weighting[1] * lossn[mask0]
#                 # multiply batch elements by weights
#                 loss[0] = lossn.mean()
#             else:
#                 loss = loss_function(y_pred, y_true)  # loss of one batch
#
#             total_loss += loss.item()  # added sum because reduction is zero and needs to be one ele to add item
#
#             y_trues = torch.cat((y_trues, y_true))
#             y_preds = torch.cat((y_preds, y_pred))
#             if to_train:
#                 # Backward:
#                 optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
#                 # optimizer.step()  # use gradients
#                 # for i in range(5):
#                     # optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
#                     # loss[i].backward(retain_graph=True)  # get gradients
#                     # optimizer.step()  # use gradients
#                 loss.backward()  # get gradients
#
#                 # Optimization step:
#                 optimizer.step()  # use gradients
#
#             # Progress bar:
#             pbar.update(1)
#
#     return total_loss, y_trues, y_preds

#
# train_loss_vec = []
# test_loss_vec = []
# val_loss_vec = []
# train_acc_vec = []
# test_acc_vec = []
# val_acc_vec = []
# for i_epoch in range(num_epochs):
#     train_loss = 0
#     test_loss = 0
#     val_loss = 0
#
#     print(f'Epoch: {i_epoch + 1}/{num_epochs}')
#     # Train set
#     train_loss, y_true_train, y_pred_train = forward_epoch(model, train_loader, loss_function, optimizer,
#                                                            train_loss,
#                                                            to_train=True, desc='Train', device=device)
#     # Test set
#     test_loss, y_true_test, y_pred_test = forward_epoch(model, test_loader, loss_function, optimizer, test_loss,
#                                                         to_train=False, desc='Test', device=device)
#     # # Validation set
#     # val_loss, y_true_val, y_pred_val = forward_epoch(sex_net, dl_val, loss_function, optimizer, val_loss,
#     #                                                  to_train=False, desc='Validation', device=gpu_0, label=label)
#
#     # Metrics:
#     train_loss = train_loss / train_ds_size  # we want to get the mean over batches.
#     test_loss = test_loss / test_ds_size
#     # val_loss = val_loss / len(dl_val)
#     train_loss_vec.append(train_loss)
#     test_loss_vec.append(test_loss)
#     # val_loss_vec.append(val_loss)
#
#     # scikit-learn computations are numpy based; thus should run on CPU and without grads.
#     train_accuracy = accuracy_score(y_true_train.cpu(),
#                                     (y_pred_train.cpu().detach() > 0.5) * 1)
#     test_accuracy = accuracy_score(y_true_test.cpu(),
#                                    (y_pred_test.cpu().detach() > 0.5) * 1)
#     # val_accuracy = accuracy_score(y_true_val.cpu(),
#     #                               (y_pred_val.cpu().detach() > 0.5) * 1)
#     train_acc_vec.append(train_accuracy)
#     test_acc_vec.append(test_accuracy)
#     # val_acc_vec.append(val_accuracy)
#
#     print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
#           test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')
# try:
#     if fn != 'None':
#         if fn[-7:] != ".pickle":
#             fn = fn + ".pickle"
#         torch.save(model.state_dict(), fn)
#         torch.save(model.state_dict(), fn[:-7]+'_opt'+fn[-7:])
#         #torch.save(sex_net, f=fn)
#         print('saved model')
# except:
#     print("didn't save")
#     pass
