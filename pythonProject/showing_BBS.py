import shutil
import json
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


# first_folder = '/home/stu16/PycharmProjects/pythonProject/thesisData/images/6/'
# dest_folder = '/home/stu16/PycharmProjects/pythonProject/thesisData/images/6_partial_data/'
#
# for i in range(10):
#     file_name = full_ds.coco.imgs[i]['file_name'][9:]
#     src = first_folder + file_name
#     dst = dest_folder + file_name
#     shutil.copy(src, dst)
#
# with open("thesisData/labels_without_multiple_labels.json") as f:
#     some_labels = json.load(f)
#     some_labels["images"] = some_labels["images"][:10]
#     some_labels["annotations"] = some_labels["annotations"][:10]
# with open("thesisData/10_samples.json", "w") as outfile:
#     json.dump(some_labels, outfile)


# min_xs = []
# min_ys = []
# max_xs = []
# max_ys = []
#
# for output in outputs:
#     bb = output["boxes"][0]
#     min_xs += [int(bb[0].item())]
#     min_ys += [int(bb[1].item())]
#     max_xs += [int(bb[2].item())]
#     max_ys += [int(bb[3].item())]
#
#
# for i in range():
#     im = torch.squeeze(images[i]).cpu().numpy()
#
#     # cv2.imshow('image', mat=im*5000)
#     cv2.rectangle(im, (min_xs[i], min_ys[i]), (max_xs[i], max_ys[i]), color=(0, 0, 255), thickness=2)
#     plt.imshow(im, cmap='gray')
#     plt.show()


def show_preds(images, predicted_labels, ground_truth_labels):
    """

    :param images: images that were inputted to the model for evaluation
    :param predicted_labels: the model's bounding box predictions for the images
    :param ground_truth_labels: ground truth bounding boxes for the images
    :return:
    """

    min_xs = []
    min_ys = []
    max_xs = []
    max_ys = []
    min_xs_gt = []
    min_ys_gt = []
    max_xs_gt = []
    max_ys_gt = []

    for output in predicted_labels:
        bb = output["boxes"][0]
        min_xs += [int(bb[0].item())]
        min_ys += [int(bb[1].item())]
        max_xs += [int(bb[2].item())]
        max_ys += [int(bb[3].item())]

    for output in ground_truth_labels:
        bb = output["boxes"][0]
        min_xs_gt += [int(bb[0].item())]
        min_ys_gt += [int(bb[1].item())]
        max_xs_gt += [int(bb[2].item())]
        max_ys_gt += [int(bb[3].item())]

    for i in range():
        im = torch.squeeze(images[i]).cpu().numpy()

        # cv2.imshow('image', mat=im*5000)
        cv2.rectangle(im, (min_xs[i], min_ys[i]), (max_xs[i], max_ys[i]), color=(0, 0, 255), thickness=2)
        cv2.rectangle(im, (min_xs_gt[i], min_ys_gt[i]), (max_xs_gt[i], max_ys_gt[i]), color=(0, 255, 255), thickness=1)

        plt.imshow(im, cmap='gray')
        plt.show()
