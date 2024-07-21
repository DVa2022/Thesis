import os
from PIL import Image
import numpy as np
import copy
import json
# import pandas as pd
import pandas as pd
import main
import pickle
import warnings
import math
import random
import sys
import torchvision
import pytorch_vision_detection.engine as engine
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import cv2
import shutil
import imageio
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Sized,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

from torch import default_generator, randperm
from torch import Tensor
from torch._utils import _accumulate
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import Sampler
from ultralytics import YOLO

T = TypeVar('T')


def convert_coco_to_yolo():
    with open("thesisData/labels2.json") as f:
        labels = json.load(f)
    img_width = 627
    img_height = 385
    for image_id, img_name in enumerate(labels['images']):
        for annotation in labels['annotations']:
            if annotation['image_id'] != image_id:
                continue
            else:
                bb = annotation['bbox']
                x = str(bb[0] / img_width + 0.5 * bb[2] / img_width)
                y = str(bb[1] / img_height + 0.5 * bb[3] / img_height)
                width = str(bb[2] / img_width)
                height = str(bb[3] / img_height)
                text_file_content = '0 ' + x + ' ' + y + ' ' + width + ' ' + height + '\n'
                text_file_name = img_name['file_name'][9:].replace('jpg', 'txt')
                with open('yolo_data/labels_I_created/' + text_file_name, 'w') as f:
                    f.write(text_file_content)
                break


def prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds):
    # the function deletes the content of the 'labels' and 'images' subdirectories of the 'train', 'val' and 'test'
    # directories and then fills the subdirectories with the labels and images that are prescribed by the train,
    # validation and test sets.

    path_start = '/workspace/PycharmProjects/pythonProject/' if 'workspace' in os.getcwd() else ''
    datasets = [train_ds, val_ds, test_ds]
    sub_sets = ['train2', 'valid2', 'test2']
    for ds_name, ds in zip(sub_sets, datasets):
        labels_directory_path = path_start + 'datasets/' + ds_name + '/labels'
        images_directory_path = path_start + 'datasets/' + ds_name + '/images'
        label_files = [file for file in os.listdir(labels_directory_path) if 'txt' in file]
        image_files = [file for file in os.listdir(images_directory_path) if 'jpg' in file]

        # emptying the labels folder
        for file in label_files:
            file_path = os.path.join(labels_directory_path, file)
            removed = os.remove(file_path)  # the variable is used to prevent printing to the console
        # emptying the images folder
        for file in image_files:
            file_path = os.path.join(images_directory_path, file)
            removed = os.remove(file_path)  # the variable is used to prevent printing to the console
        # filling the labels and images folders
        for sub_ds_index in ds.indices:
            file_name = full_ds.coco.dataset['images'][sub_ds_index]['file_name'][9:]
            shutil.copyfile(path_start + 'datasets/images/' + file_name, images_directory_path + '/' + file_name)
            file_name = file_name.replace('jpg', 'txt')
            shutil.copyfile(path_start + 'datasets/labels/' + file_name, labels_directory_path + '/' + file_name)
        print('Finished preparing the ' + ds_name + ' data.')


def crop_data():
    # crops all the images in images_path and saves the cropped images in target_path
    # generally target_path should contain the dataset

    images_path = 'C:/Users/David/PycharmProjects/previous_data_versions/6_original'  # path of the folder from which the images are taken
    target_path = 'C:/Users/David/PycharmProjects/thesisData/images/6'  # path of the folder where the new images are saved
    all_images_list = os.listdir(images_path)
    for image_path in all_images_list:
        complete_image_path = os.path.join(images_path, image_path)
        image = Image.open(complete_image_path).convert('L')  # open image and convert to grayscale
        image = np.array(image)[8:393, 147:774]  # cropping the image
        Image.fromarray(image).save(os.path.join(target_path, image_path))


def adapt_annotations_to_data_cropping():
    # This is the cropping: [8:393, 147:774]

    f = open("C:/Users/David/PycharmProjects/thesisData/labels.json")
    labels = json.load(f)
    for annotation in labels["annotations"]:  # composed of xmin, ymin, width, height
        annotation["bbox"][0] = annotation["bbox"][0] - 147  # xmin of the bounding box
        annotation["bbox"][1] = annotation["bbox"][1] - 8  # ymin of the bounding box

    labels["categories"][0]['name'] = 'Background'
    labels["categories"] += [{'id': 1,
                              'name': 'Pleura'}]

    with open("C:/Users/David/PycharmProjects/thesisData/labels2.json", "w") as outfile:
        json.dump(labels, outfile)


def Faster_RCNN_three_annotations_per_image_correction():
    # the function creates a list of dictionaries which represents the annotations in the Faster RCNN format.
    # the new list has three annotations for each image and the old list has a varying number between 1 and 3.
    # the function is used, so I can input Faster RCNN with more than one annotation per image, whereas the original
    # list (labels2) can't be inputted to the network because of the varying number of annotations per image.
    # In the new list images which originally had only 1 annotation have three annotations but two of them are with
    # sort of empty bounding boxes.
    # labels2 should be in the Fast RCNN format to begin with.
    # labels2 is the original labels file after the bounding boxes but with xmin and ymin shifted after the images were cropped.

    f = open("C:/Users/David/PycharmProjects/thesisData/labels2.json")
    labels2 = json.load(f)
    i = 0
    annotations_with_triple_labels = []
    while i < len(labels2["annotations"]):
        annotation = copy.deepcopy(labels2["annotations"][i])
        annotations_with_triple_labels += [annotation]
        annotations_with_triple_labels[-1]["id"] = i * 3
        if i < len(labels2["annotations"]) - 2:  # almost always this is the case
            next_equals = annotation['image_id'] == labels2["annotations"][i + 1]['image_id']
            second_next_equals = annotation['image_id'] == labels2["annotations"][i + 2]['image_id']
            if next_equals and second_next_equals:  # if an image has three annotations
                annotations_with_triple_labels += [copy.deepcopy(labels2["annotations"][i + 1])]
                annotations_with_triple_labels[-1]["id"] = i * 3 + 1
                annotations_with_triple_labels += [labels2["annotations"][i + 2]]
                annotations_with_triple_labels[-1]["id"] = i * 3 + 2
                i += 3
            elif next_equals and not second_next_equals:  # if an image has two annotations
                annotations_with_triple_labels += [copy.deepcopy(labels2["annotations"][i + 1])]
                annotations_with_triple_labels += [copy.deepcopy(labels2["annotations"][i + 1])]
                annotations_with_triple_labels[-1]["bbox"] = [1, 1, 1,
                                                              1]  # the value of one was chosen so all boxes have positive dimensions
                annotations_with_triple_labels[-1]["area"] = 0
                annotations_with_triple_labels[-1]["id"] = i * 3 + 1
                i += 2
            elif not next_equals and not second_next_equals:  # if an image has only one annotation
                annotations_with_triple_labels += [copy.deepcopy(annotation)]
                annotations_with_triple_labels += [copy.deepcopy(annotation)]
                annotations_with_triple_labels[-1]["bbox"] = [1, 1, 1, 1]
                annotations_with_triple_labels[-1]["area"] = 0
                annotations_with_triple_labels[-1]["id"] = i * 3 + 1
                annotations_with_triple_labels[-2]["bbox"] = [1, 1, 1, 1]
                annotations_with_triple_labels[-2]["area"] = 0
                annotations_with_triple_labels[-1]["id"] = i * 3 + 2
                i += 1
            else:  # this never happens
                print("Something is wrong. i=" + str(i))
                break
        # if we've reached the penultimate annotation there's no point in checking if it equals the next two
        # non-existent annotations. We should check only the next annotation
        elif i == len(labels2["annotations"]) - 2:
            next_equals = annotation['image_id'] == labels2["annotations"][i + 1]['image_id']
            if next_equals:  # if an image has two annotations
                annotations_with_triple_labels += [copy.deepcopy(labels2["annotations"][i + 1])]
                annotations_with_triple_labels += [copy.deepcopy(labels2["annotations"][i + 1])]
                annotations_with_triple_labels[-1]["bbox"] = [1, 1, 1, 1]
                annotations_with_triple_labels[-1]["area"] = 0
                annotations_with_triple_labels[-1]["id"] = i * 3 + 1
                i += 2
            else:  # if an image has only one annotation
                annotations_with_triple_labels += [copy.deepcopy(annotation)]
                annotations_with_triple_labels += [copy.deepcopy(annotation)]
                annotations_with_triple_labels[-1]["bbox"] = [1, 1, 1, 1]
                annotations_with_triple_labels[-1]["area"] = 0
                annotations_with_triple_labels[-1]["id"] = i * 3 + 1
                annotations_with_triple_labels[-2]["bbox"] = [1, 1, 1, 1]
                annotations_with_triple_labels[-2]["area"] = 0
                annotations_with_triple_labels[-1]["id"] = i * 3 + 2
                i += 1
        # if we've reached the last annotation there's no point in checking if it equals the next two non-existent annotations
        elif i == len(labels2["annotations"]) - 1:  #
            annotations_with_triple_labels += [copy.deepcopy(annotation)]
            annotations_with_triple_labels += [copy.deepcopy(annotation)]
            annotations_with_triple_labels[-1]["bbox"] = [1, 1, 1, 1]
            annotations_with_triple_labels[-1]["area"] = 0
            annotations_with_triple_labels[-1]["id"] = i * 3 + 1
            annotations_with_triple_labels[-2]["bbox"] = [1, 1, 1, 1]
            annotations_with_triple_labels[-2]["area"] = 0
            annotations_with_triple_labels[-1]["id"] = i * 3 + 2

    # correcting the indices of the labels. Ideally the while loop would do it, but it doesn't for some reason
    for i, annotation in enumerate(annotations_with_triple_labels):
        annotation["id"] = i

    # if it's a real object and BB, give it the category of an object (which is 1. 0 is for the background)
    for i, annotation in enumerate(annotations_with_triple_labels):
        if annotation["area"] != 0:
            annotation["category_id"] = 1

    labels_with_triple_labels = labels2
    labels_with_triple_labels["annotations"] = annotations_with_triple_labels
    with open("C:/Users/David/PycharmProjects/thesisData/labels_with_triple_labels.json", "w") as outfile:
        json.dump(labels_with_triple_labels, outfile)

    return labels_with_triple_labels


def Faster_RCNN_single_annotation_per_image_correction():
    # the function creates a list of dictionaries which represents the annotations in the Faster RCNN format.
    # the new list has only one annotation per image and the old list has a varying number between 1 and 3.
    # the function is used, so I can input Faster RCNN with only one annotation per image, whereas the original
    # list (labels2) can't be inputted to the network because of the varying number of annotations per image.
    # to do the correction each image that has more than one annotation gets some of its annotations deleted.
    # labels2 should be in the Fast RCNN format to begin with.
    # labels2 is the original labels file after the bounding boxes but with xmin and ymin shifted after the images were cropped.

    f = open("C:/Users/David/PycharmProjects/thesisData/labels2.json")
    labels2 = json.load(f)
    # actually I should have created a copy that is independent of the original variable, but I didn't:
    labels_without_multiple_labels = labels2
    for i, annotation in enumerate(labels_without_multiple_labels["annotations"]):
        if labels_without_multiple_labels["annotations"][i]['image_id'] == \
                labels_without_multiple_labels["annotations"][i + 1]['image_id']:
            del labels_without_multiple_labels["annotations"][i + 1]
        if i < len(labels_without_multiple_labels["annotations"]) - 1:
            if labels_without_multiple_labels["annotations"][i]['image_id'] == \
                    labels_without_multiple_labels["annotations"][i + 1]['image_id']:
                del labels_without_multiple_labels["annotations"][i + 1]

    for i, annotation in enumerate(labels_without_multiple_labels["annotations"]):
        # if it's a real object and BB, give it the category of an object (which is 1. 0 is for the background)
        annotation["category_id"] = 1
        annotation["id"] = annotation["image_id"]

    with open("C:/Users/David/PycharmProjects/thesisData/labels_without_multiple_labels.json", "w") as outfile:
        json.dump(labels_without_multiple_labels, outfile)

    return labels_without_multiple_labels


def not_all_taggable_videos_are_tagged_adaptation(images_path):
    # in my laptop images_path is 'C:\Users\David\PycharmProjects\thesisData\images\6'

    if 'David' in os.getcwd():
        IDlist_path = 'C:/Users/David/PycharmProjects/pythonProject1/IDlist.npy'
    else:
        IDlist_path = 'IDlist.npy'

    IDlist = np.load(IDlist_path)

    names_of_suitable_videos, tagged_frames_list, _, _, _, _ = main.get_list_of_suitable_videos_in_drive()
    names_of_suitable_videos = names_of_suitable_videos[:, 2]
    taggable_video_names = []  # of size 2444
    for name in names_of_suitable_videos:
        taggable_video_names += [name[7:-4]]
    tagged_video_names = []

    tagged_video_names2 = os.listdir(images_path)  # of size 2427
    for name in tagged_video_names2:
        tagged_video_names += [name[28:38]]  # remaining with only the number in the file names

    # some not obligatory sanity check:
    tagged_videos_counter = 0
    for vid in tagged_video_names:
        if vid in taggable_video_names:
            tagged_videos_counter += 1

    tagged_videos_mask = []
    for vid in taggable_video_names:
        if vid in tagged_video_names:
            tagged_videos_mask += [True]
        else:
            tagged_videos_mask += [False]

    tagged_IDlist = IDlist[tagged_videos_mask]
    unique_patient_IDs, unique_patient_IDs_count = np.unique(tagged_IDlist, return_counts=True)

    return unique_patient_IDs, unique_patient_IDs_count, tagged_IDlist, tagged_videos_mask


def find_best_data_split(images_path, train_share=0.6, val_share=0.25, test_share=0.15, num_of_iterations=20000):
    # in my laptop images_path is 'C:\Users\David\PycharmProjects\thesisData\images\6'
    # train_share, val_share and test_share are the training split percentages

    full_ds_size = len(os.listdir(images_path))  # 2427, for example
    best_split_error = 100000  # arbitrarily large number
    ideal_training_set_size = int(full_ds_size * train_share)
    ideal_val_set_size = int(full_ds_size * val_share)
    ideal_test_set_size = int(full_ds_size * test_share)

    unique_patient_IDs, unique_patient_IDs_count, tagged_IDlist, tagged_videos_mask = not_all_taggable_videos_are_tagged_adaptation(
        images_path)
    num_of_patients = unique_patient_IDs_count.size  # 339
    random_split_mask = np.zeros(unique_patient_IDs_count.size, dtype=int)  # 0 marks the training set patients
    random_split_mask[:int(num_of_patients * val_share)] = 1  # 1 marks the validation set patients
    random_split_mask[int(num_of_patients * val_share):int(
        num_of_patients * (val_share + test_share))] = 2  # 2 marks the test set patients

    for i in range(num_of_iterations):
        np.random.seed(int(random.uniform(0, 1) * 100))
        # np.random.seed(i)
        np.random.shuffle(random_split_mask)
        training_set_size = np.sum(unique_patient_IDs_count[random_split_mask == 0])
        val_set_size = np.sum(unique_patient_IDs_count[random_split_mask == 1])
        test_set_size = np.sum(unique_patient_IDs_count[random_split_mask == 2])
        split_error = abs(training_set_size - ideal_training_set_size) + abs(val_set_size - ideal_val_set_size) + abs(
            test_set_size - ideal_test_set_size)
        if split_error < best_split_error:
            best_split_error = split_error
            best_random_split_mask = random_split_mask.copy()
            if best_split_error == 1:
                print(best_random_split_mask[:25])
                break

    print('Best_split_error: ' + str(best_split_error) + ' was found at iteration number ' + str(i))

    patient_names_train = unique_patient_IDs[best_random_split_mask == 0]
    patient_names_val = unique_patient_IDs[best_random_split_mask == 1]
    patient_names_test = unique_patient_IDs[best_random_split_mask == 2]

    indices_of_data_split = tagged_IDlist.tolist()  # a list in the form of [1, 1, 0, 2, 0, 0, 0, 1, ...]. Its size is as the number of videos
    for i, patient in enumerate(indices_of_data_split):
        if indices_of_data_split[i] in patient_names_train:
            indices_of_data_split[i] = 0
        elif indices_of_data_split[i] in patient_names_val:
            indices_of_data_split[i] = 1
        elif indices_of_data_split[i] in patient_names_test:
            indices_of_data_split[i] = 2
        else:
            print("There's a patient outside of the patients list!")
            break

    indices_of_data_split = np.array(indices_of_data_split)
    train_indices = np.where(indices_of_data_split == 0)[0]  # [0] is used because it's a tuple
    val_indices = np.where(indices_of_data_split == 1)[0]
    test_indices = np.where(indices_of_data_split == 2)[0]
    indices = np.concatenate((train_indices, val_indices, test_indices))
    indices = torch.tensor(indices)
    dataset_sizes = [train_indices.size, val_indices.size, test_indices.size]

    return indices, dataset_sizes, tagged_videos_mask


def random_split(indices, dataset_sizes, dataset: Dataset[T], lengths: Sequence[Union[int, float]],
                 generator: Optional[torch.Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = indices.tolist()
    # indices2 = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    lengths = dataset_sizes
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def find_best_unique_split(images_path, prev_indices, train_share=0.6, val_share=0.25, test_share=0.15):
    for i in range(50):
        found_equal_indices = False
        indices, dataset_sizes, tagged_videos_mask = find_best_data_split(images_path=images_path,
                                                                          train_share=train_share,
                                                                          val_share=val_share,
                                                                          test_share=test_share)

        # checking if the new indices are unique. If not, find new indices. If yes, return them.
        new_indices = indices[:50]
        # by single run I mean that these are the indices that were used to train a single model
        for single_run_indices in prev_indices:
            if torch.equal(single_run_indices, new_indices):
                # if the new indices are equal to at least one row in prev_indices, there's no need to check more rows
                found_equal_indices = True
                break
        if found_equal_indices == False:  # if new_indices is unique, there's no need to create new splits
            prev_indices = torch.vstack((prev_indices, new_indices))
            return indices, dataset_sizes, tagged_videos_mask, prev_indices

    print("Failed 50 times in creating a unique data split.")
    return indices, dataset_sizes, tagged_videos_mask, prev_indices


class MyRandomWeightedSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, sex_mask, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

        self.sex_mask = sex_mask
        if np.sum(sex_mask == 'M') > np.sum(sex_mask == 'F'):
            self.num_of_over_represented = np.sum(sex_mask == 'M')
            self.num_of_under_represented = np.sum(sex_mask == 'F')
            self.over_represented_group = 'M'
            self.under_represented_group = 'F'
            self.over_represented_group_indices = np.where(sex_mask == 'M')[0]  # [0] is used because it's a tuple
            self.under_represented_group_indices = np.where(sex_mask == 'F')[0]
        else:
            self.num_of_over_represented = np.sum(sex_mask == 'F')
            self.num_of_under_represented = np.sum(sex_mask == 'M')
            self.over_represented_group = 'F'
            self.under_represented_group = 'M'
            self.over_represented_group_indices = np.where(sex_mask == 'F')[0]  # [0] is used because it's a tuple
            self.under_represented_group_indices = np.where(sex_mask == 'M')[0]
        self.num_of_replaced = self.num_of_over_represented - int(len(self.data_source) / 2)

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                ordered_indices = torch.arange(n)
                smaller_set_size = min(self.over_represented_group_indices.size,
                                       self.under_represented_group_indices.size)
                rand_index = torch.randint(high=smaller_set_size - int(self.num_of_replaced), size=(1,),
                                           dtype=torch.int64, generator=self.generator)
                cropped_over_represented_indices = self.over_represented_group_indices[
                                                   rand_index:rand_index + self.num_of_replaced]

                if self.under_represented_group_indices.size < cropped_over_represented_indices.size:
                    X = cropped_over_represented_indices.size
                    tensor = torch.tensor(self.under_represented_group_indices)
                    cropped_under_represented_indices = torch.cat([tensor] * ((len(tensor) + X) // len(tensor)))[:X]
                else:
                    cropped_under_represented_indices = self.under_represented_group_indices[
                                                        rand_index:rand_index + self.num_of_replaced]

                ordered_indices[cropped_over_represented_indices] = torch.tensor(cropped_under_represented_indices)
                random_order_indices = torch.randperm(n, generator=generator)
                shuffled_indices = ordered_indices[random_order_indices]

                # under_represented_indices = np.where(self.sex_mask == self.under_represented_group)
                # cropped_over_represented_indices = over_represented_indices[:self.num_of_replaced].tolist()
                # cropped_under_represented_indices = under_represented_indices[:self.num_of_replaced]
                # random_order_indices[cropped_over_represented_indices] = cropped_under_represented_indices
                yield from shuffled_indices.tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


def one_epoch(model, optimizer, dl, epoch, tb_writer, loss_fn, to_train=True, FRCNN_flag=False):
    total_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dl):
        if to_train:
            # Every data instance is an input + label pair
            inputs, labels = data

            # Make predictions for this batch
            if FRCNN_flag:
                labels2 = []
                for i in range(labels['boxes'].size()[0]):
                    dict = {}
                    dict['boxes'] = labels['boxes'][i]
                    dict['labels'] = labels['labels'][i]
                    dict['image_id'] = labels['image_id'][i]
                    dict['area'] = labels['area'][i]
                    dict['iscrowd'] = labels['iscrowd'][i]
                    labels2 += [dict]
                outputs = model(inputs, labels2)  # TODO outputs includes the losses. Also the total loss?
            else:
                outputs = model(inputs)
                # Compute the loss
                loss = loss_fn(outputs, labels)

            # Gather data and report
            total_loss += loss.item()
        else:  # if it's validation, don't use gradients
            with torch.no_grad():
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

        if to_train:
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # compute gradients of the loss
            loss.backward()

            # Adjust learning weights
            optimizer.step()

        if i % 1000 == 999:
            last_loss = total_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(dl) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            total_loss = 0.

    return total_loss


def log_variables(experiment_dir, experiment_name, optimizer=None, train_sampler=None, lr_scheduler=None,
                  batch_size=None, num_classes=None,
                  num_epochs=None, device=None, num_workers=None, data_split=None, max_epochs_without_improvement=None,
                  transforms_train=None, train_ds_size=None, val_ds_size=None, test_ds_size=None, path2json=None):
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(experiment_dir + '/log.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        print('experiment_name: ' + experiment_name + '\n')
        print('path2json: ' + path2json + '\n')
        print('optimizer:\n')
        print(optimizer)
        print('\n')
        print(train_sampler)
        print(lr_scheduler)
        print('\n')
        print('batch_size: ' + str(batch_size))
        print('num_classes: ' + str(num_classes))
        print('num_epochs: ' + str(num_epochs))
        print('device:')
        print(device)
        print('num_workers: ' + str(num_workers))
        print('data_split: ' + str(data_split))
        print('max_epochs_without_improvement: ' + str(max_epochs_without_improvement))
        print('\n')
        print('transforms_train:')
        print(transforms_train)
        print('train_ds_size: ' + str(train_ds_size))
        print('val_ds_size: ' + str(val_ds_size))
        print('test_ds_size: ' + str(test_ds_size))

        sys.stdout = original_stdout  # Reset the standard output to its original value


def FASTER_RCNN_inference(val_loader, device, num_classes=2, get_IoUs=False,
                          experiment_dir='/home/stu16/PycharmProjects/pythonProject/',
                          model_path='model_20230926_111219_5', resnet_flag=True, tiny_convnext=None):
    if resnet_flag:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        import torchvision.models
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator

        backbone = torchvision.models.convnext_large(weights="DEFAULT").features
        backbone.out_channels = 768 if tiny_convnext else 1536  # 1536 for large. 1024 for base
        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 24, 32),),
            aspect_ratios=((0.5, 0.25, 1 / 6),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    model.to(device)
    model.load_state_dict(torch.load(experiment_dir + '/' + model_path))
    model.eval()
    evaluation_output = engine.evaluate(model, val_loader, device=device, mode="Validation", get_IoUs=get_IoUs,
                                        show_preds=False, model_path=model_path)
    print(evaluation_output)
    # creating a plot of the distribution of the IoUs of the validation set videos
    if get_IoUs:
        IoUs, precision, recall = evaluation_output
        np.save(experiment_dir + "/precision_" + str(np.array(precision))[:6], np.array(precision))
        np.save(experiment_dir + "/recall_" + str(np.array(recall))[:6], np.array(recall))
        mean_IoU = np.array(IoUs).mean()
        print('Mean IoU is ', mean_IoU)
        hist = plt.hist(IoUs, bins=20)
        plt.xlabel("IoU")
        plt.ylabel("Number of videos")
        plt.title("IoU Distribution. Mean IoU: " + str(mean_IoU)[:5])
        plt.savefig(experiment_dir + '/all_video_IoUs.png')

    return IoUs, precision, recall  # if get_IoUs==True, it's the IoUs. Otherwise, it's the evaluator


def boxes_shared_area(boxA, boxB, IoU=False):
    """
    **The boxes should be in the format of [xmin, ymin, xmax, ymax]**
    :param IoU: a flag. If it's True it means that the IoU should be returned
    :param boxA: a numpy array representing the bounding box with the most confident score in an image
    :param boxB: a numpy array representing one of the bounding boxes whose isn't the most confident
    :return: The intersection area between two bounding boxes divided by the area of the first bounding box.
    """

    # getting the coordinates that represent the intersection area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection between the two BBs
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # area of boxA
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

    if IoU:
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        val_of_IoU = interArea / (boxAArea + boxBArea - interArea)
        return val_of_IoU
    else:
        shared_area_percentage = interArea / boxAArea  # between 0 and 1
        return shared_area_percentage


def show_preds(images, predicted_labels, ground_truth_labels, save=True, model_path=''):
    """

    :param model_path: The name of the folder that should be created for the detections
    :param save: whether to save the comparison of ground truth and predicted BBs or to show them without saving
    :param images: images that were inputted to the model for evaluation
    :param predicted_labels: the model's bounding box predictions for the images
    :param ground_truth_labels: ground truth bounding boxes for the images
    :return:
    """

    if 'detections' not in os.listdir(os.getcwd()):
        os.mkdir('detections')
    if model_path not in os.listdir('detections'):
        os.mkdir('detections/' + model_path)

    min_xs = []
    min_ys = []
    max_xs = []
    max_ys = []
    min_xs_gt = []
    min_ys_gt = []
    max_xs_gt = []
    max_ys_gt = []

    for output in predicted_labels:
        if torch.numel(output["boxes"]) != 0:
            bb = output["boxes"][0]
            min_xs += [int(bb[0].item())]
            min_ys += [int(bb[1].item())]
            max_xs += [int(bb[2].item())]
            max_ys += [int(bb[3].item())]
        else:
            min_xs += [0]
            min_ys += [0]
            max_xs += [1]
            max_ys += [1]

    for output in ground_truth_labels:
        bb = output["boxes"][0]
        min_xs_gt += [int(bb[0].item())]
        min_ys_gt += [int(bb[1].item())]
        max_xs_gt += [int(bb[2].item())]
        max_ys_gt += [int(bb[3].item())]

    for i in range(len(images)):
        # the next line can be used when I don't do any normalization to the images when training
        # im = torch.squeeze(images[i]).cpu().numpy() * 255
        # normalizing the pixels between 0 and 255
        im = torch.squeeze(images[0]).cpu().numpy()
        im = im - im.min()
        im = im / im.max() * 255
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        # cv2.imshow('image', mat=im*5000)
        cv2.rectangle(im, (min_xs[i], min_ys[i]), (max_xs[i], max_ys[i]), color=(0, 0, 255), thickness=1)  # red color
        cv2.rectangle(im, (min_xs_gt[i], min_ys_gt[i]), (max_xs_gt[i], max_ys_gt[i]), color=(255, 0, 0), thickness=1)

        image_id = ground_truth_labels[i]['image_id'].item()

        if save:  # if I wish to save the detections
            cv2.imwrite('detections/' + model_path + '/' + str(image_id) + '.png', im)
        else:  # if I wish to show them im pycharm and not save them
            plt.imshow(im, cmap='gray')
            plt.title('Image ID: ' + str(image_id))
            plt.show()

    # triple_labels = True
    # if triple_labels:
    #     all_confident_BBs = []
    #     for output in predicted_labels:  # for each image
    #         confident_predictions = output["scores"] > 0.9  # if the model is relatively confident about the BB
    #         # TODO consider changing the next two line if I input the network frames that don't have objects
    #         if (output["scores"] > 1).sum().item() == 0:  # if there aren't any boxes with a score above 0.9
    #             confident_predictions[0] = True  # then consider the most confident box to be a correct detection
    #         all_confident_BBs += [output["boxes"][confident_predictions]]
    #
    #     all_gt_boxes = []
    #     for output in ground_truth_labels:
    #         not_fake_boxes = output["area"] != 0
    #         all_gt_boxes += [output["boxes"][not_fake_boxes]]
    #
    #     for i in range(len(images)):
    #         im = torch.squeeze(images[i]).cpu().numpy() * 255
    #         im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    #
    #         predicted_bbs = all_confident_BBs[i].cpu().numpy()
    #         if predicted_bbs.size > 4:
    #             BBs_to_delete = []
    #             for k in range(1, predicted_bbs.shape[0]):
    #                 shared_area_percentage = boxes_shared_area(boxA=predicted_bbs[0], boxB=predicted_bbs[k])
    #                 if shared_area_percentage > 0.2:
    #                     BBs_to_delete += [k]
    #             for bb_to_delete in BBs_to_delete:
    #                 indices_of_bbs_to_be_deleted = torch.arange(1, all_confident_BBs[i].shape[0] + 1) != bb_to_delete
    #                 all_confident_BBs[i] = all_confident_BBs[i][~indices_of_bbs_to_be_deleted, :]
    #
    #         for bb in all_confident_BBs[i].cpu().numpy():
    #             min_xs = int(bb[0])
    #             min_ys = int(bb[1])
    #             max_xs = int(bb[2])
    #             max_ys = int(bb[3])
    #             cv2.rectangle(im, (min_xs, min_ys), (max_xs, max_ys), color=(0, 0, 255), thickness=1)  # blue color
    #
    #         for bb in all_gt_boxes[i].cpu().numpy():
    #             min_xs_gt = int(bb[0])
    #             min_ys_gt = int(bb[1])
    #             max_xs_gt = int(bb[2])
    #             max_ys_gt = int(bb[3])
    #             cv2.rectangle(im, (min_xs_gt, min_ys_gt), (max_xs_gt, max_ys_gt), color=(255, 0, 0), thickness=1)
    #
    #         image_id = ground_truth_labels[i]['image_id'].item()
    #
    # if save:  # if I wish to save the detections
    #     cv2.imwrite('detections/' + model_path + '/' + str(image_id) + '.png', im)
    # else:  # if I wish to show them im pycharm and not save them
    #     plt.imshow(im, cmap='gray')
    #     plt.title('Image ID: ' + str(image_id))
    #     plt.show()


def save_dls(experiment_dir, val_loader=None, test_loader=None, train_loader=None):
    if val_loader != None:
        file = open(experiment_dir + '/val_loader.p', 'wb')
        pickle.dump(val_loader, file)
        file.close()

    if test_loader != None:
        file = open(experiment_dir + '/test_loader.p', 'wb')
        pickle.dump(test_loader, file)
        file.close()

    if train_loader != None:
        file = open(experiment_dir + '/train_loader.p', 'wb')
        pickle.dump(train_loader, file)
        file.close()


def match_model_to_sample(train_sets, dicom_name):
    """

    :param train_sets: A list of train sets. Each train set is a 1D numpy array. Each train set corresponds to a model
    that was trained using the train set.
    :param dicom_name: The name of the video/dicom file.
    :return: The name of the first model in the list of models that was trained using the video called dicom_name.
    """
    model_name = None
    for k, train_set in enumerate(train_sets):  # looping over the models
        if np.sum(train_set == dicom_name) == 1:  # if the video was used to train the model
            model_name = 'fourth_take_no_dfl_extra_large_model' + str(k + 17)
            return model_name
    # if no model was found, it's the case of a video that appears in the dataset files with a name that is shorter by
    # one letter than the true name of the DICOM
    if model_name is None:
        model_name = match_model_to_sample(train_sets, dicom_name[:-1])
        return model_name

    # code for creating 'train_sets'
    # train_sets += [np.load(
    #     '/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model17/train_set.npy')]
    # train_sets += [np.load(
    #     '/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model18/train_set.npy')]
    # train_sets += [np.load(
    #     '/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model19/train_set.npy')]
    # train_sets += [np.load(
    #     '/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model20/train_set.npy')]
    # train_sets += [np.load(
    #     '/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model21/train_set.npy')]

    # with open('train_sets_for_tracking_list', 'wb') as fp:
    #     pickle.dump(train_sets, fp)
    # with open('train_sets_for_tracking_list', 'rb') as fp:
    #     sample_list = pickle.load(fp)

    # for j in range(0, 5):
    #     for i, file in enumerate(train_sets[j]):
    #         train_sets[j][i] = file[21:-7]


def one_video_object_tracking(annotated_frame, model_dir, dicom_name,
                              not_annotated_numpy_frames_folder=None, save_images=False, save_raw_frames=True,
                              save_video=False, save_annotation=True, last_frame=None, pause_at_annotated_frame=False, numpy_videos_folder='numpy_videos'):
    """
    In this function inference and tracking are done on an annotated frame and on the sequential not annotated frames
    of a video. The tracking results can be saved as a video and/or individual images and/or YOLO-style annotation
    text files.

    :param save_raw_frames: Whether to save the raw frames and create a folder for them.
    :param pause_at_annotated_frame: Whether to make a video with a pause when the annotated frame is reached.
    :param dicom_name: The name of the original dicom file.
    :param save_annotation: Whether to save annotation files based on the tracking results or not.
    :param annotated_frame: The number of the frame that was annotated.
    :param last_frame: The last frame in the video that we wish to do inference/tracking on. This argument is needed
                       only if separated_numpy_files_flag=True.
    :param model_dir: The folder where the trained model is found. It's recommended that the model was trained on a
                      frame from the video that tracking will be done on.
    :param not_annotated_numpy_frames_folder: Folder of the not annotated numpy frames.
    :param save_images: The saving folder of the images that contain the original frame with the found bounding
                               box. If no folder is given there won't be any saving.
    :param save_video: Whether to create a video with painted bounding boxes on the found object or not.
    :return: None
    """

    path_start = '/workspace/PycharmProjects/pythonProject/' if 'workspace' in os.getcwd() else ''  # if we're on the DGX we need to change the path

    # making a folder if it doesn't exist
    if save_images and dicom_name not in os.listdir(path_start + 'datasets/tracked frames'):
        os.mkdir(path_start + 'datasets/tracked frames/' + dicom_name)

    # making a folder if it doesn't exist
    if save_annotation and dicom_name not in os.listdir(path_start + 'datasets/tracking labels'):
        os.mkdir(path_start + 'datasets/tracking labels/' + dicom_name)

    # making a folder if it doesn't exist
    if save_raw_frames and dicom_name not in os.listdir('datasets/raw frames'):
        os.mkdir('datasets/raw frames/' + dicom_name)

    separated_numpy_files_flag = False
    # an array of all relevant frames to be inputted to the tracker and to have a rectangle painted on
    if separated_numpy_files_flag:  # if the frames are in separate files
        all_frames = []
        for i in range(annotated_frame, last_frame):
            all_frames += [np.load(
                path_start + 'datasets/single frames/' + not_annotated_numpy_frames_folder + '/' + str(i) + '.npy',
                allow_pickle=True)]
        all_frames = np.array(all_frames)
    else:
        all_frames = np.load(path_start + 'datasets/' + numpy_videos_folder + '/' + dicom_name + '.npy')

    # creating the image files on which the tracking/inference will be done
    if save_raw_frames:
        if separated_numpy_files_flag:
            i = annotated_frame
            for frame in all_frames:
                im = Image.fromarray(frame)
                im.save('datasets/raw frames/' + dicom_name + '/' + dicom_name + '_' + str(i + 1) + '.jpg')
                i += 1
        else:
            for i, frame in enumerate(all_frames):
                im = Image.fromarray(frame)
                im.save('datasets/raw frames/' + dicom_name + '/' + dicom_name + '_' + str(i + 1) + '.jpg')

    # image_folder = '/home/stu16/PycharmProjects/pythonProject/datasets/tracking_results/'
    # images = [img2 for img2 in os.listdir(image_folder) if img2.endswith(".png")]
    # images = [int(num.split('.')[0]) for num in images]
    # images.sort()
    # images = [str(file_num) + '.png' for file_num in images]

    # 1 for looping forward in time and -1 for back in time (from the labeled frame to the first one)
    step_options = [1, -1]
    loop_end_options = [len(all_frames), -1]
    images_w_rectangle = []  # frames with bounding boxes drawn on them
    for step, loop_end in zip(step_options, loop_end_options):
        model_path_start = path_start if numpy_videos_folder == 'numpy_videos' else '/workspace/'
        model = YOLO(
            model_path_start + 'runs/detect/' + model_dir + '/weights/best.pt')  # initializing the model every loop iteration
        for i in range(annotated_frame, loop_end, step):  # Read a frame from the video
            frame = all_frames[i]
            not_annotated_frame_path = path_start + 'datasets/raw frames/' + dicom_name + '/' + dicom_name + '_' + str(
                i + 1) + '.jpg'
            results = model.track(not_annotated_frame_path, persist=True, conf=0.05)
            box = results[0].boxes.xyxy
            # if there aren't any detections or if more than one detection is found for the same object
            # (if there's more than one detection, the wrong detection might be with a higher confidence level so
            # changing the NMS (Non-Max Suppression) value (which is the iou argument of the track method) won't help)
            if torch.numel(box) == 0 or torch.numel(box) > 4:
                if save_video:
                    im = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    images_w_rectangle += [im]
                if torch.numel(box) == 0:
                    shutil.copyfile(not_annotated_frame_path,
                                    path_start + 'datasets/background_frames/' + dicom_name + '_' + str(i + 1) + '.jpg')
            else:
                # if results[0].boxes.conf < 0.25:
                #     print(dicom_name, i, results[0].boxes.conf)
                if save_images or save_video:
                    im = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    images_w_rectangle += [im]
                    cv2.rectangle(im, (int(box[0, 0].item()), int(box[0, 1].item())),
                                  (int(box[0, 2].item()), int(box[0, 3].item())),
                                  color=(0, 0, 255), thickness=1)  # red color
                if save_images and i != annotated_frame:
                    cv2.imwrite(path_start + 'datasets/tracked frames/' + dicom_name + '/' + dicom_name + '_' + str(
                        i + 1) + '.png', im)

                # TODO make sure that the annotated frame isn't saved as an image
                # TODO add the option of a pause at the annotated frame when creating a video

                # saving annotation text files in YOLO format + there's no need to create a label for the manually labeled frame
                if save_annotation and i != annotated_frame:
                    obj_class = results[0].boxes.cls.int().item()  # the predicted class
                    box_for_saving = results[0].boxes.xywhn[
                        0].cpu().numpy()  # the predicted bounding box in YOLO format
                    # creating the annotation text to be saved
                    annotation_text = str(obj_class)
                    for val in box_for_saving:
                        annotation_text += ' ' + str(val)
                    annotation_text += '\n'
                    # saving the annotation text
                    text_file = open(
                        path_start + 'datasets/tracking labels/' + dicom_name + '/' + dicom_name + '_' + str(
                            i + 1) + '.txt', "w")
                    n = text_file.write(annotation_text)
                    text_file.close()

    if save_video:
        if not separated_numpy_files_flag:
            # arranging the frames from the (temporally) first one to the last one
            length = len(images_w_rectangle)
            images_w_rectangle = images_w_rectangle[length - annotated_frame:][::-1] + images_w_rectangle[
                                                                                       :length - annotated_frame]
        imageio.v2.mimsave(path_start + 'datasets/tracking videos/' + dicom_name + '.gif',
                           images_w_rectangle, duration=8)

        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # use ('I','4','2','0') for avi files which are better
        # fps = 10.0
        # width = 627
        # height = 385
        # video_saving_folder = '/datasets/tracking videos/' + dicom_name + '.mp4'
        # video = cv2.VideoWriter(video_saving_folder, fourcc, fps, (width, height))
        # for frame in images_w_rectangle:
        #     video.write(frame)
        # video.release()


def inference_video_creation(model_dir, dicom_name):
    """
    In this function inference is done on all frames of a video, and a video with predicted bounding boxes is saved.

    :param dicom_name: The name of the original dicom file.
    :param model_dir: The folder where the trained model is found. It's recommended that inference is done on frames
                      from a video that has no frames in the training set of the model.
    :return: None
    """

    on_dgx = 'workspace' in os.getcwd()
    datasets_path = '/workspace/PycharmProjects/pythonProject/datasets/' if on_dgx else 'datasets/'
    models_path = '/workspace/runs/detect/' if on_dgx else 'runs/detect/'

    # an array of all relevant frames to be inputted to the model
    all_frames = np.load(datasets_path + 'numpy_videos/' + dicom_name + '.npy')

    images_w_rectangle = []  # frames with bounding boxes drawn on them
    model = YOLO(models_path + model_dir + '/weights/best.pt')
    for i, frame in enumerate(all_frames):  # Read a frame from the video
        vid_file_name_beginning = dicom_name + '_' if on_dgx else ''
        frame_path = datasets_path + 'raw frames/' + dicom_name + '/' + vid_file_name_beginning + str(
            i + 1 if on_dgx else i) + '.jpg'
        # 'iou' is for non-max suppression. 'max_det' is max number of allowed detections in a video
        results = model(frame_path, iou=0.3, max_det=3)
        box = results[0].boxes.xyxy
        # if there aren't any detections or if more than one detection is found for the same object
        # (if there's more than one detection, the wrong detection might be with a higher confidence level so
        # changing the NMS (Non-Max Suppression) value (which is the iou argument of the track method) won't help)
        im = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        images_w_rectangle += [im]
        if not (torch.numel(box) == 0 or torch.numel(box) > 4):
            cv2.rectangle(im, (int(box[0, 0].item()), int(box[0, 1].item())),
                          (int(box[0, 2].item()), int(box[0, 3].item())),
                          color=(0, 0, 255), thickness=1)  # red color

    # saving the video
    imageio.v2.mimsave(datasets_path + 'inference videos/' + dicom_name + '.gif', images_w_rectangle, duration=8)


def multiple_inference_videos_creation(model_dir, vid_names, inference_on_the_inferred=False):
    """
    This function does inference on several videos using a YOLOv8 model. The videos and the model weights are chosen by
    the user. It doesn't do inference on videos that already got inference done on, unless specified otherwise.
     It saves videos showing the inference results (the bounding boxes drawn on the frames).

    :param inference_on_the_inferred: Whether to do inference on videos that already have inference videos.
    :param model_dir: The folder where the trained model is found. It's recommended that inference is done on frames
                      from a video that has no frames in the training set of the model.
    :param vid_names: The names of the videos that the model will do inference on. They should be in the format of
                     'hashed_#'.
    :return: None
    """

    on_dgx = 'workspace' in os.getcwd()
    datasets_path = '/workspace/PycharmProjects/pythonProject/datasets/' if on_dgx else 'datasets/'
    # videos that this function was already used on
    vids_with_inference_already_done = os.listdir(datasets_path + 'inference videos')
    # removing the file extensions
    vids_with_inference_already_done = [vid[:vid.find('.gif')] for vid in vids_with_inference_already_done]

    # vid_names = os.listdir(vid_names_folder)
    # vid_names = [vid_name.split('_')[3] + '_' + vid_name.split('_')[4] for vid_name in vid_names]

    count = 0
    for vid_name in vid_names:
        print(count)
        # if it's a video that inference was already done on, continue, unless specified otherwise
        if not inference_on_the_inferred and vid_name in vids_with_inference_already_done:
            continue
        inference_video_creation(model_dir, vid_name)
        count += 1


def get_tagged_frame_index(desired_vid_names):
    """

    :param desired_vid_names: A list of the names of the videos for which we wish to find what was the tagged frame.
    The names should come in the format of 'hashed_...' without a file extension.
    :return: A list of the indices of the frames that were tagged. The order of videos is the same as the order of
    videos in desired_vid_names.
    """

    on_dgx = 'workspace' in os.getcwd()
    if on_dgx:
        csv_path = '/workspace/all_data_270222_2.csv'
    else:
        username = os.getcwd().split('/')[2]  # either 'stu16' or 'davidva'
        csv_path = '/home/' + username + '/all_data_270222_2.csv'

    patient_data = pd.read_csv(csv_path)
    all_vid_names = patient_data['Patient_ID']
    all_tagged_frames = patient_data['Tagged_Frame']
    desired_tagged_frame_indices = []
    for vid_name in desired_vid_names:
        video_index = all_vid_names.index[all_vid_names == vid_name][0]
        desired_tagged_frame_indices += [all_tagged_frames[video_index]]

    return desired_tagged_frame_indices


def semisupervision_prep():
    on_dgx = 'workspace' in os.getcwd()
    vid_names_folder = '/workspace/PycharmProjects/pythonProject/datasets/labels' if on_dgx else '/datasets/labels'
    train_sets_folder = '/workspace/PycharmProjects/pythonProject/runs/detect/' if on_dgx else '/runs/detect/'
    tracked_vids_folder = '/workspace/PycharmProjects/pythonProject/datasets/raw frames' if on_dgx else '/datasets/raw frames'

    vid_names = os.listdir(vid_names_folder)
    vid_names = [vid_name.split('_')[3] + '_' + vid_name.split('_')[4] for vid_name in vid_names]
    annotated_frame_numbers = get_tagged_frame_index(vid_names)
    # adjusting the indexing from one-based to zero-based, to prevent one_video_object_tracking() from trying to reach
    # frames that are out of bounds for a certain numpy array
    annotated_frame_numbers = [num - 1 for num in annotated_frame_numbers]

    train_sets = []
    for i in range(17, 22):
        train_sets += [np.load(train_sets_folder + 'fourth_take_no_dfl_extra_large_model' + str(i) + '/train_set.npy')]

    for j in range(0, 5):
        for i, file in enumerate(train_sets[j]):
            train_sets[j][i] = file[21:-7]
    # with open('train_sets_for_tracking_list', 'wb') as fp:
    #     pickle.dump(train_sets, fp)
    # with open('train_sets_for_tracking_list', 'rb') as fp:
    #     sample_list = pickle.load(fp)

    tracked_vids = os.listdir(tracked_vids_folder)
    count = 0
    for vid_name, annotated_frame_number in zip(vid_names, annotated_frame_numbers):
        print(count)
        if vid_name in tracked_vids and vid_name != 'hashed_1235855420':  # if it's a video that object tracking was already done on
            continue
        model_dir = match_model_to_sample(train_sets, vid_name)
        one_video_object_tracking(annotated_frame_number, model_dir, vid_name, save_images=True)
        count += 1


def semisupervised_training_set_prep(training_order_dict, max_num_of_frames_from_each_vid):

    # if we're on the DGX we need to change the path
    train_set_folder = '/workspace/PycharmProjects/pythonProject/datasets/train2' if 'workspace' in os.getcwd() else 'datasets/train2'
    labels_source_dir =  '/workspace/PycharmProjects/pythonProject/datasets/tracking labels/' if 'workspace' in os.getcwd() else 'datasets/tracking labels/'
    raw_frames_source_dir = '/workspace/PycharmProjects/pythonProject/datasets/raw frames/' if 'workspace' in os.getcwd() else 'datasets/raw frames/'
    
    curr_valid_data = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/valid2/labels')
    curr_valid_data = [sample for sample in curr_valid_data if 'hashed' in sample]
    curr_test_data = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/test2/labels')
    curr_test_data = [sample for sample in curr_test_data if 'hashed' in sample]
    curr_valid_data = [sample[sample.find('hashed'):] for sample in curr_valid_data]
    curr_test_data = [sample[sample.find('hashed'):] for sample in curr_test_data]
    curr_valid_data = ["_".join(sample.split('_')[:2]) for sample in curr_valid_data]
    curr_test_data = ["_".join(sample.split('_')[:2]) for sample in curr_test_data]
    curr_valid_data = np.array(curr_valid_data)
    curr_test_data = np.array(curr_test_data)

    if 'original_train_set' not in os.listdir(train_set_folder):
        original_training_set = os.listdir(train_set_folder + '/labels')
        # write list to binary file
        with open(train_set_folder + '/original_training_set', 'wb') as fp:
            pickle.dump(original_training_set, fp)
    else:
        with open(train_set_folder + '/original_training_set', 'rb') as fp:
            original_training_set = pickle.load(fp)

    # making the names of the labels and training images in the format of "hashed_#_#"
    dir_options = ['images', 'labels']
    for dir in dir_options:
        for sample in os.listdir(train_set_folder + '/' + dir):
            if len(sample) > 30:  # if it's smaller than 30 the name of the file is already in the right short format
                string_split = sample.split('_')
                new_sample_name = string_split[3] + '_' + string_split[4] + '_' + string_split[5]
                os.rename(train_set_folder + '/' + dir + '/' + sample, train_set_folder + '/' + dir + '/' + new_sample_name)

    # creating a dictionary that maps between video name (in the format of hashed_###) to patient ID
    with open("id_dict.json", "r") as outfile:
        id_dict_annotated = json.load(outfile)
    with (open("id_dict_unannotated.pkl", "rb")) as openfile:
        id_dict_unannotated = pickle.load(openfile)
    id_dict_unannotated = {key[:key.find('.dcm')]: value for key, value in zip(id_dict_unannotated.keys(), id_dict_unannotated.values())}  # removing '.dcm' from the video names
    id_dict_annotated.update(id_dict_unannotated)
    id_dict = id_dict_annotated  # changing the name, so it's not misleading, because it's actually a dictionary of annotated and unannotated videos
    
    val_ids = []
    for vid in curr_valid_data:
        val_ids += [id_dict_annotated[vid]]
    val_unique_ids = list(set(val_ids))
    test_ids = []
    for vid in curr_test_data:
        test_ids += [id_dict_annotated[vid]]
    test_unique_ids = list(set(test_ids))
    val_and_test_unique_ids = val_unique_ids + test_unique_ids

    # moving the automatically created data to the training folder
    for vid_name, training_order in zip(training_order_dict.keys(), training_order_dict.values()):
        # if the video is in the validation or test sets, or if it belongs to a patient that has videos in the validation or test sets,
        # don't move frames of the video to the training set 
        if vid_name in curr_valid_data or vid_name in curr_test_data or id_dict[vid_name] in val_and_test_unique_ids:
            continue
        max_possible_num_of_training_frames = min(len(training_order), max_num_of_frames_from_each_vid)
        # whether to move the first frame (in training_order) to the training set or not. If it's a frame from the video that
        # the Ichilov staff didn't annotate, it should be moved
        starting_frame_to_be_moved = 0 if vid_name not in id_dict_unannotated.keys() else 1
        
        frames_to_be_moved = training_order[starting_frame_to_be_moved:max_possible_num_of_training_frames]
        one_vid_labeled_frames = os.listdir(labels_source_dir + vid_name)
        for frame_to_be_moved in frames_to_be_moved:
            file_name = vid_name + '_' + str(frame_to_be_moved) + '.'
            if file_name + 'txt' in one_vid_labeled_frames:
                shutil.copy(labels_source_dir + vid_name + '/' + file_name + 'txt', train_set_folder + '/labels/' + file_name + 'txt')
                shutil.copy(raw_frames_source_dir + vid_name + '/' + file_name + 'jpg', train_set_folder + '/images/' + file_name + 'jpg')


def dataset_preparation_for_unannotated_scans(skip_raw_frames_creation=False):
    """
    This function deals with the videos that weren't annotated by the Ichilov staff. These videos didn't even get an
    annotation for one frame. This function uses a trained YOLOv8x model, that was trained on around 44000 frames, and
    does inference on the frames of the videos. There are about 2300 videos. The frame of each video was cropped, so it
    could be inputted to the model. The videos are in the format of a numpy file. When the model finishes inference on
    a video, it checks the number of the frame in the video that got the highest prediction confidence. The frame with
    the highest confidence will be the frame with which the object tracking will be done.

    The function saves a dictionary that maps between video name to frame with the highest prediction probability.
    It also saves a dictionary that maps between the video name to the number of frames in the video.
    It also save a dictionary between the video name and the video's training order.

    Lastly, it creates the artificial annotations with object tracking for all unannotated videos.

    :return: None.
    """

    model = YOLO("/workspace/runs/detect/semi-supervised_XL_model_training-series3_32-fold_increase_44119/weights/best.pt")
    
    on_dgx = 'workspace' in os.getcwd()
    vid_names_folder = '/workspace/PycharmProjects/pythonProject/datasets/numpy_videos_not_originally_annotated' if on_dgx else '/datasets/numpy_videos_not_originally_annotated'
    # train_sets_folder = '/workspace/PycharmProjects/pythonProject/runs/detect/' if on_dgx else '/runs/detect/'
    raw_frames_folder = '/workspace/PycharmProjects/pythonProject/datasets/raw frames' if on_dgx else '/datasets/raw frames'

    vid_names = os.listdir(vid_names_folder)
    vid_names = [vid_name[:vid_name.find('.npy')] for vid_name in vid_names]
    raw_frames_subfolders = os.listdir(raw_frames_folder)
    most_confident_prediction_indices = {}
    vid_lengths = {}
    training_order_dict_unannotated = {}
    
    if skip_raw_frames_creation:
        # loading the dictionaries
        with open("/workspace/PycharmProjects/pythonProject/vid_lengths_unannotated_scans.json") as outfile:
            vid_lengths = json.load(outfile)
        with open("/workspace/training_order_dict_unannotated_scans.json") as outfile:
            training_order_dict_unannotated = json.load(outfile)
        with open("/workspace/labeled_frames_unannotated_scans.json") as outfile:
            most_confident_prediction_indices = json.load(outfile)
    else:
        for vid_name in vid_names:
            # making a folder if it doesn't exist
            if vid_name not in raw_frames_subfolders:
                os.mkdir(raw_frames_folder + '/' + vid_name)
            # creating the image files on which the tracking will be done
            # note that "i + 1" is used to convert back to one-based indexing from zero-based indexing
            all_frames = np.load(vid_names_folder + '/' + vid_name + '.npy')
            for i, frame in enumerate(all_frames):
                im = Image.fromarray(frame)
                im.save(raw_frames_folder + '/' + vid_name + '/' + vid_name + '_' + str(i + 1) + '.jpg')
    
            frames_dir = raw_frames_folder + '/' + vid_name + '/'
    
            # discovering what's the frame on which the model is most confident when doing inference
            confidences = []
            vid_lengths[vid_name] = all_frames.shape[0]
            for frame_num in range(vid_lengths[vid_name]):
                result = model(frames_dir + vid_name + '_' + str(frame_num + 1) + '.jpg')
                if torch.numel(result[0].boxes.xyxy) != 0:  # if there are detections
                    confidence = result[0].boxes.conf[0].item()
                else:  # if there aren't any detections, there isn't a confidence value so we'll simply assign zero
                    confidence = 0
                confidences += [confidence]
    
            most_confident_prediction_indices[vid_name] = int(np.argmax(np.array(confidences)) + 1)
    
            training_order_dict_unannotated[vid_name] = create_one_vid_train_order(vid_lengths[vid_name], most_confident_prediction_indices[vid_name])

        # saving the dictionaries
        with open("/workspace/PycharmProjects/pythonProject/vid_lengths_unannotated_scans.json", "w") as outfile:
            json.dump(vid_lengths, outfile)
        with open("/workspace/training_order_dict_unannotated_scans.json", "w") as outfile:
            json.dump(training_order_dict_unannotated, outfile)
        with open("/workspace/labeled_frames_unannotated_scans.json", "w") as outfile:
            json.dump(most_confident_prediction_indices, outfile)
    

    # creating the artificial annotations with object tracking
    model_dir = 'semi-supervised_XL_model_training-series3_32-fold_increase_44119'
    tracked_vids_folder = '/workspace/PycharmProjects/pythonProject/datasets/tracking labels' if on_dgx else '/datasets/tracking labels'
    tracked_vids = os.listdir(tracked_vids_folder)
    count = 0
    for vid_name in vid_names:
        print(count)
        count += 1
        #  if vid_name in tracked_vids:  # if it's a video that object tracking was already done on
        #      continue
        save_images = count % 100 == 0
        one_video_object_tracking(most_confident_prediction_indices[vid_name] - 1, model_dir, vid_name, save_images=save_images, save_raw_frames=False, numpy_videos_folder='numpy_videos_not_originally_annotated')


def create_all_vids_train_order(vid_names=None, vid_lengths=None, vid_labeled_frames=None, use_unannotated_scans=False):
    """
    The function creates a dictionary that maps between video names to numpy arrays containing the order of frames for
    a semi-supervised training on some/all frames. If the file already exists the function simply loads it instead of
    creating it.

    :param use_unannotated_scans: Whether to use the scans that weren't annotated by the Ichilov staff.
    :param vid_names: a list/1D array of the names of all videos in the dataset.
    :param vid_lengths: a dictionary that maps between video names and their corresponding number of frames.
    :param vids_labeled_frames: a list/1D array of the numbers of the labeled frames in the dataset videos.
    :return: a dictionary that maps between video names to numpy arrays containing the order of frames for
             a semi-supervised training on some/all frames.
    """

    # the lines that should be run to create the function inputs:
    # vid_names_folder = 'datasets/labels'
    # vid_names = os.listdir(vid_names_folder)
    # vid_names = [vid_name.split('_')[3] + '_' + vid_name.split('_')[4] for vid_name in vid_names]
    # vid_labeled_frames = get_tagged_frame_index(vid_names)
    # with open("vid_lengths.json", "r") as f:
    #     vid_lengths = json.load(f)

    if 'training_order_dict.json' in os.listdir(os.getcwd()):
        # opening the dictionary
        with open("training_order_dict.json", "r") as outfile:
            training_order_dict = json.load(outfile)
    else:
        training_order_dict = {}

        for vid_name, labeled_frame_index in zip(vid_names, vid_labeled_frames):
            training_order_dict[vid_name] = create_one_vid_train_order(vid_lengths[vid_name], labeled_frame_index)

        # saving the dictionary
        with open("training_order_dict.json", "w") as outfile:
            json.dump(training_order_dict, outfile)
    
    if use_unannotated_scans:
        with open("training_order_dict_unannotated_scans.json", "r") as outfile:
            training_order_dict_unannotated_scans = json.load(outfile)
            training_order_dict.update(training_order_dict_unannotated_scans)
    
    return training_order_dict
    

def create_one_vid_train_order(num_of_frames, labeled_frame_index):
    """
    The function is used to get the order of the frames that are used for training. The first frame is the labeled frame.
    The second frame is the temporally farthest frame from the labeled frame. The third frame is the temporally farthest
    frame from the two previous frames etc. The reasoning is the frames that are farthest from each other are likely to
    be relatively different, thus if we train with only some of the frames it makes more sense to choose the framse
    that have the biggest variability among them.

    :param num_of_frames: an integer representing the number of frames in the video.
    :param labeled_frame_index: an integer representing the index of the frame that was labeled in the video.
    :return: a 1D numpy array with the size of labeled_frame_index. Its first cell is equal to labeled_frame_index.
             The cells of the array are the order of indices of the frames. If we wish to train a model with X frames,
             we should use the first X cells of the array.

    """

    arr = np.zeros(num_of_frames, dtype=int)
    arr[labeled_frame_index - 1] = 1  # "-1" for conversion from one-indexind to zero-indexing
    add_value = 2
    while np.count_nonzero(arr == 0) > 0:  # stops if there are no cells with the value of zero
        indices_of_non_zeros = np.ravel(np.argwhere(arr > 0))
        # example value of dist: [16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0  1  2  3]
        dist = np.array([min(abs(i - indices_of_non_zeros)) for i in range(len(arr))])
        idx = np.argmax(dist)
        arr[idx] = add_value
        add_value += 1

    # example value of arr: [ 2 11  7 12  4 13  8 14  3 15  9 16  5 17 10 18  1 19 20  6]
    # example of output: [16,  0,  8,  4, 12, 19,  2,  6, 10, 14,  1,  3,  5,  7,  9, 11, 13, 15, 17, 18]

    return np.argsort(arr).tolist()


# def create_semi_supervised_train_set(all_vids_train_order, max_num_of_frames_added_to_training):
# 
#     dir_options = ['images', 'labels']
#     tracked_vids_dir_options = ['raw frames', 'tracking labels']
#     current_training_frames = [file[:-4] for file in os.listdir('datasets/train/labels')]  # frames without '.txt'
#     for dir, tracked_vids_dir in zip(dir_options, tracked_vids_dir_options):
#         destination_dir = 'datasets/train/' + dir
#         for sample in tqdm(os.listdir(destination_dir)):
#             string_split = sample.split('_')
#             new_sample_name = string_split[0] + '_' + string_split[1]  # the original name without the frame number
#             source_dir = 'datasets/' + tracked_vids_dir + '/' + new_sample_name
#             if max_num_of_frames_added_to_training > all_vids_train_order[new_sample_name].size - 1:
#                 max_num_of_frames_added_to_training = all_vids_train_order[new_sample_name].size - 1
#             # the first element is the labeled frame, so it's skipped:
#             frames_to_be_copied = all_vids_train_order[new_sample_name][1:max_num_of_frames_added_to_training + 1]
#             for frame_num in frames_to_be_copied:
#                 file_name = new_sample_name + '_' + frame_num
#                 if file_name in current_training_frames:
#                     continue
#                 else:
#                     # TODO check if the names are okay
#                     shutil.copy(source_dir, destination_dir)


#  The following function was commented out because of some importing problem that happened when you
#  run yolo_experiment.py. The solution is moving the function to yolo_experiment.py.

# def find_split_that_covers_rest_of_data_and_train(images_path, data_split, full_ds, batch_size, device, num_workers):
#
#     # you use this function if you have a few models with weights that can be used, plus these models have numpy
#     # train_set files in their folders. This function combines the train_set files to a list of numpy files and
#     # calculates the number of unique samples that these files cover. Later it finds a new data split that would cover
#     # the samples that the previous models weren't trained on. This way all samples have at least on model that was
#     # trained on them. When this happens you can use one_video_object_tracking() on all videos, knowing that there's
#     # at least one model that's relatively good on the frame (from the video) that it was trained on.
#     # Later the function trains a model on the found split.
#
#     mAPs13 = []
#     print('no_dfl experiment, extra large model')
#     train_sets = []
#     for i in range(17, 21):  # these numbers correspond to the previous models that have weights and train_set files
#         train_sets += [np.load(
#             '/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model' + str(
#                 i) + '/train_set.npy')]
#     train_sets = np.concatenate(train_sets)  # combining the train_set files to one big files
#     best_size = np.unique(train_sets).size  # number of samples in all previously trained models
#
#     for i in range(1):  # only one model is created, assuming that its data split will cover the rest of the samples
#         seed = int(torch.empty((), dtype=torch.int64).random_().item())
#         generator = torch.Generator()
#         generator.manual_seed(seed)
#         for j in range(100):  # trying 100 times to find a good split
#             indices, dataset_sizes, _, prev_indices = find_best_unique_split(images_path=images_path,
#                                                                                    train_share=data_split[0],
#                                                                                    val_share=data_split[1],
#                                                                                    test_share=data_split[2],
#                                                                                    prev_indices=prev_indices)
#             train_ds, val_ds, test_ds = random_split(indices, dataset_sizes, full_ds, data_split, generator)
#             curr_train_set = np.array([sample['file_name'][9:].replace('jpg', 'txt') for sample in train_ds.dataset.coco.imgs.values()])[
#                 train_ds.indices]
#             # the number of the unique splits in the combined train_sets
#             two_set_uniques_size = np.unique(np.concatenate((train_sets, curr_train_set))).size
#             if two_set_uniques_size > best_size:  # if there are more unique samples, it's better
#                 best_data_split = (train_ds, val_ds, test_ds)
#                 best_size = two_set_uniques_size
#                 print(j, two_set_uniques_size)
#         train_ds, val_ds, test_ds = best_data_split  # ideally it would cover all samples (2427)
#         prepare_images_and_labels_for_yolo(train_ds, val_ds, test_ds, full_ds)
#
#         model = YOLO("yolov8x.pt")
#
#         # setting 'rect=True' gives: WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
#         # setting 'imgsz=627' gives: WARNING ⚠️ imgsz=[627] must be multiple of max stride 32, updating to [640]
#
#         prev_dirs = [dir for dir in os.listdir('runs/detect')]
#         results = model.train(data='data.yaml', epochs=90, imgsz=640,
#                               patience=14,
#                               batch=batch_size, cache=True, device=device, workers=num_workers, name='fourth_take_no_dfl_extra_large_model',
#                               optimizer='SGD', lr0=0.0016, lrf=0.025, degrees=0, translate=0,
#                               scale=0, mosaic=0, mixup=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, dfl=0)
#         new_dirs = [dir for dir in os.listdir('runs/detect')]
#         newest_dir = [dir for dir in new_dirs if not dir in prev_dirs][0]
#         yolo_experiment.save_datasets(run_dir=newest_dir)
#         model = YOLO('runs/detect/' + newest_dir + '/weights/best.pt')
#         yolo_experiment.get_val_results(model, saving_dir=newest_dir, valid_or_test='valid')
#         yolo_experiment.get_val_results(model, saving_dir=newest_dir, valid_or_test='test')
#
#         # # delete all previous weight files if the last model is better than all previous ones
#         # if i >= 2 and results.box.ap[0] > max(mAPs13):
#         #     for dir in prev_dirs:
#         #         if 'weights' in os.listdir(os.path.join('runs/detect', dir)):
#         #             shutil.rmtree('runs/detect/' + dir + '/weights')
#
#         mAPs13 += [results.box.ap[0]]
