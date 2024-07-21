import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from ultralytics import YOLO
from transformers import SamProcessor
import torch.nn as nn
import csv

import pdb
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import json
join = os.path.join
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from skimage.metrics import hausdorff_distance
from torch.optim import Adam
from tqdm import tqdm
from statistics import mean
import monai

from PIL import Image
import torch
import utils
from contextlib import redirect_stdout


def draw_bb_on_pred(medsam_pred, prompt, shrink_factor=4):
    """
    This function overlays a bounding box on a predicted segmentation mask.

    Args:
        medsam_pred (torch.Tensor): The predicted segmentation mask as a tensor, 
                                     typically with shape (H, W).
        prompt (torch.Tensor): The bounding box coordinates tensor with shape (4,). 
                               The elements represent (x1, y1, x2, y2).
        shrink_factor (int, optional): A factor by which to shrink the bounding box 
                                         before drawing it on the mask. Defaults to 4.

    Returns:
        torch.Tensor: The modified segmentation mask tensor with the bounding box drawn on/around it.

    Notes:
        * This function assumes the bounding box coordinates are in the format 
          (x1, y1, x2, y2), where:
            * x1 and y1 represent the top-left corner of the bounding box.
            * x2 and y2 represent the bottom-right corner of the bounding box.
        * The function modifies the input medsam_pred tensor directly by setting 
          specific elements to a value to represent the bounding box lines.
    """
    
    # adjusting the coordinates of the prompt
    prompt = prompt / shrink_factor
    
    x1 = int(prompt[0].item())
    x2 = int(prompt[2].item())
    y1 = int(prompt[1].item())
    y2 = int(prompt[3].item())
    medsam_pred = medsam_pred.squeeze().cpu()
    
    # drawing the bounding box
    medsam_pred[y1:y2 + 1, x1] = 0.5
    medsam_pred[y1:y2 + 1, x2] = 0.5
    medsam_pred[y1, x1:x2 + 1] = 0.5
    medsam_pred[y2, x1:x2 + 1] = 0.5
    
    return medsam_pred


def visualize_segmentation(gt_mask, pred_mask):
  """
  This function visualizes the segmentation results using numpy functions 
  for efficiency.

  Args:
      gt_mask: Ground truth segmentation mask (2D numpy array).
      pred_mask: Predicted segmentation mask (2D numpy array).

  Returns:
      A new RGB image (3D numpy array) visualizing the segmentation results.
  """
  
  # Check if shapes are compatible
  if gt_mask.shape != pred_mask.shape:
    raise ValueError("Ground truth and predicted masks must have the same shape.")

  # Black for correct foreground (gt=1, pred=1)
  # [255, 0, 0], Red for wrongly predicted background (gt=1, pred=0)
  # [0, 255, 255], Yellow for wrongly predicted foreground (gt=0, pred=1)
  # [0, 255, 0], Green for correctly predicted background (gt=0, pred=0)

  # Combine conditions for each color using boolean indexing
  correct_foreground = (gt_mask == 1) & (pred_mask == 1)
  wrong_background = (gt_mask == 1) & (pred_mask == 0)
  wrong_foreground = (gt_mask == 0) & (pred_mask == 1)  
  comparison = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
  comparison[correct_foreground] = [0, 255, 0]
  comparison[wrong_background] = [255, 0, 0]
  comparison[wrong_foreground] = [0, 255, 255]
  
  return comparison



def medsam_inference(medsam_model, img_embed, box_1024, H, W, tensor_output=False):

    """
    This function performs some of the forward pass of the MedSam model. It's based on script from the MedSAM git.
    
    Args:
        medsam_model (torch.nn.Module): The MedSam model instance.
        img_embed (torch.Tensor): The embedded image tensor, typically with shape (B, 256, 64, 64).
        box_1024 (torch.Tensor): The bounding box coordinates tensor with shape (B, 4) in format (x1, y1, x2, y2).
        H (int): The height of the desired output segmentation mask.
        W (int): The width of the desired output segmentation mask.
        tensor_output (bool, optional): If True, the function returns the segmentation mask as a tensor. 
                                         Defaults to False (numpy array).
    
    Returns:
        tuple: A tuple containing:
            - segmentation_mask (torch.Tensor or np.ndarray): The predicted segmentation mask with shape (B, 1, H, W).
            - low_res_pred (torch.Tensor): The low-resolution logits before upsampling, with shape (B, 1, H, W).
    """

    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    
    # changing the shape of the prediction
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu()
    
    # a threshold of 0.5 is used to create the boolean segmentation prediction
    if tensor_output:
        medsam_seg = (low_res_pred > 0.5).to(torch.int32)
    else:
        low_res_pred = low_res_pred.numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    
    return medsam_seg, low_res_pred


class SAM_dataset(Dataset):

    """
    PyTorch dataset class for preparing ultrasound images and annotations for MedSam.
  
    This class loads ultrasound images and corresponding segmentation annotations, 
    performs preprocessing steps like cropping and normalization, and generates 
    inputs compatible with the MedSam model. It also utilizes YOLOv8 models 
    (if provided) to generate bounding boxes around objects of interest in the 
    ultrasound images.
  
    """

    def __init__(self, images_dir, annotations_dir, processor, bb_enlargement_factor=23, YOLO_models=None, transforms=None, subimage_input=False):
        
        '''
            Args:
        images_dir (str): Path to the directory containing ultrasound images.
        annotations_dir (str): Path to the directory containing segmentation annotations.
        processor (MedSamProcessor): A MedSam processor instance. The processor is used for the creation of the output of the _getitem_ function.
        bb_enlargement_factor (int, optional): The factor by which to enlarge the bounding box during cropping. Defaults to 23. It determines the number of not-cropped out pixels around the bounding box.
        YOLO_models (list of YOLOv8 models, optional): A list of YOLOv8 models for object detection. If None, YOLO is not used.
        transforms (torchvision.transforms.v2, optional): PyTorch transformations for image augmentation. If None, basic scaling and conversion to float32 are applied.
        subimage_input (bool, optional): Whether to input the model an image that was cropped from the original image that is found in the dataset folder or to use the original image.
                                         The cropped images are cropped around the bounding box that YOLOv8 finds. Defaults to False.
        '''
        
        self.images_dir = images_dir
        annotation_list = os.listdir(annotations_dir)

        # creating a dictionary that converts from file names such as 'hashed_1233708898_43.png' to
        # '00331209-DCM_29_A671_hashed_1233708898_43.jpg'
        short_to_long_img_name_conversion_dict = {}
        for file in annotation_list:
            for file2 in os.listdir(images_dir):
                if file.replace('png', '') in file2:
                    short_to_long_img_name_conversion_dict[file] = file2
                    break

        # applying the mapping that the mentioned dictionary does
        self.image_paths = [os.path.join(images_dir, short_to_long_img_name_conversion_dict[path]) for path in
                            annotation_list]
        self.annotations_dir = annotations_dir
        # creating a list of *full* paths of the annotations
        self.annotation_paths = [os.path.join(annotations_dir, path) for path in annotation_list]
        
        self.YOLO_models = YOLO_models
        self.processor = processor

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self.length = len(os.listdir(annotations_dir))
        
        self.subimage_input = subimage_input
        self.bb_enlargement_factor = bb_enlargement_factor

        # training sets of the YOLOv8 models. Each training set is different because of the random data split
        yolo_train_set1 = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/train/labels')
        yolo_train_set2 = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/train2/labels')
        yolo_train_set3 = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/train3/labels')
        self.yolo_train_set1 = np.unique(np.array(["_".join(sample.split('_')[:2]) for sample in yolo_train_set1]))
        self.yolo_train_set2 = np.unique(np.array(["_".join(sample.split('_')[:2]) for sample in yolo_train_set2]))
        self.yolo_train_set3 = np.unique(np.array(["_".join(sample.split('_')[:2]) for sample in yolo_train_set3]))

    def __len__(self):
        return self.length

    def match_yolo_to_image(self, annotation_path):
        '''
        This function matches a trained YOLOv8 model to an image/annotation. It does so only if the model wasn't trained on the image.
        The function returns the number of the YOLOv8 model (there are a few trained models).
        If there aren't any models that weren't trained on the image -1 is returned. 
        '''

        file_name_in_short_format = "_".join(
            annotation_path[annotation_path.find('hashed'):].split('_')[:2])  # in the format of 'hashed_#'

        # checking if there's a model that didn't have the image in its training set
        # if so, return the index of the model
        if file_name_in_short_format not in self.yolo_train_set1:
            return 0
        elif file_name_in_short_format not in self.yolo_train_set2:
            return 1
        elif file_name_in_short_format not in self.yolo_train_set3:
            return 2
        else:
            return -1
    
    def move_files(self):
        '''
        this function was used only once. It was used to find and move images that don't have any YOLOv8 that was trained on them.
        '''
        
        import shutil
        
        for index in range(10000):  # 10000 is an arbitrarily large number
            img_path = self.image_paths[index]
            annotation_path = self.annotation_paths[index]
            
            file_name_in_short_format = "_".join(
            annotation_path[annotation_path.find('hashed'):].split('_')[:2])  # in the format of 'hashed_#'

            # checking if there's a model that didn't have the image in its training set
            # if so, return the index of the model
            if file_name_in_short_format not in self.yolo_train_set1:
                continue
            elif file_name_in_short_format not in self.yolo_train_set2:
                continue
            elif file_name_in_short_format not in self.yolo_train_set3:
                continue
            else:
                shutil.move(annotation_path, annotation_path.replace('segmentations', 'segmentations without models that werent trained on them'))


    def crop_image_and_prompt(self, img, prompt, annotation, W, H, blacken_out_of_bb=False):
        """
        This function crops the annotation image and the ultrasound image so that by creating a square centered around
        the bounding box prompt. It also changes the coordinates of the prompt based on the cropping.
        To

        :param img: a pytorch 3D tensor (batch X H X W) representing the original ultrasound frame
        :param prompt: a pytorch tensor representing the bounding box prompt created by YOLOv8 and used by MedSAM.
        :param annotation: a pytorch 3D tensor (batch X H X W) representing the original annotation image
        :param W: an int representing the width of the original image and annotation.
        :param H: an int representing the height of the original image and annotation.
        :param blacken_out_of_bb: a boolean variable indicating whether the pixels in the ultrasound image that are
                                  outside the prompt should be zeroed.
        :return: In short, it returns a modified version of the input parameters.
        """

        bb_enlargement_factor = self.bb_enlargement_factor  # the units are the number of pixels
        
        if torch.numel(prompt) == 0:  # this happens when YOLOv8 finds zero objects
            x_1 = W / 2 - max(bb_enlargement_factor, 8)
            x_2 = W / 2 + max(bb_enlargement_factor, 8)
            y_1 = H / 2 - max(bb_enlargement_factor, 8)
            y_2 = H / 2 + max(bb_enlargement_factor, 8)
        else:  # this is the more common case
            x_1 = prompt[0].item()
            x_2 = prompt[2].item()
            y_1 = prompt[1].item()
            y_2 = prompt[3].item()

        # center coordinates and shape of the bounding box
        x_center = (x_1 + x_2) / 2
        y_center = (y_1 + y_2) / 2
        bb_width = round(x_2 - x_1)
        bb_height = round(y_2 - y_1)

        # deciding what's the size of the new image and annotation
        subimage_W_or_H = bb_width if bb_width >= bb_height else bb_height
        # deciding the ideal location of the four corners of the new image and annotation withing the original image and annotation
        new_x1 = round(x_center) - bb_enlargement_factor - 0.5 * subimage_W_or_H
        new_x2 = round(x_center) + bb_enlargement_factor + 0.5 * subimage_W_or_H
        new_y1 = round(y_center) - bb_enlargement_factor - 0.5 * subimage_W_or_H
        new_y2 = round(y_center) + bb_enlargement_factor + 0.5 * subimage_W_or_H

        lower_y_for_cropping = round(new_y2)
        upper_y_for_cropping = round(new_y1)
        right_x_for_cropping = round(new_x2)
        left_x_for_cropping = round(new_x1)
        upper_padding_size = 0
        lower_padding_size = 0
        right_padding_size = 0
        left_padding_size = 0

        # if one of the four corners of the new image and annotation are outside of the original image and annotation, the coordinates should be changed
        if new_y2 > H - 1:
            lower_y_for_cropping = H - 1
            lower_padding_size = round(new_y2 - lower_y_for_cropping)
        if new_y1 < 0:
            upper_y_for_cropping = 0
            upper_padding_size = round(upper_y_for_cropping - new_y1)
        if new_x2 > W - 1:
            right_x_for_cropping = W - 1
            right_padding_size = round(right_x_for_cropping - new_x2)
        if new_x1 < 0:
            left_x_for_cropping = 0
            left_padding_size = round(left_x_for_cropping - new_x1)

        # cropping so the images have a square shape
        img = img[0, upper_y_for_cropping:lower_y_for_cropping, left_x_for_cropping:right_x_for_cropping]
        annotation = annotation[0, upper_y_for_cropping:lower_y_for_cropping, left_x_for_cropping:right_x_for_cropping]
        
        # if you wish to nullify all pixels that are outside of the bounding box, use this
        if blacken_out_of_bb:
            square_bb_upper_y_diff = round(y_1 - upper_y_for_cropping)
            square_bb_left_x_diff = round(x_1 - left_x_for_cropping)

            img[:square_bb_upper_y_diff, :] = 0
            img[square_bb_upper_y_diff + bb_height:, :] = 0
            img[:, :square_bb_left_x_diff] = 0
            img[:, square_bb_left_x_diff + bb_width:] = 0

        # padding with zeros so the images have a square shape, if they don't already have a square shape
        # usually the padding is of 0 pixels, meaning that usually it does nothing
        padding = (left_padding_size, right_padding_size, upper_padding_size, lower_padding_size)
        img = nn.functional.pad(img, padding)
        annotation = nn.functional.pad(annotation, padding).unsqueeze(0)

        transform = v2.ToPILImage()
        # converting the image to PIL, so we can convert it to RGB easily
        annotation_H, annotation_W = annotation.size()[1], annotation.size()[2]
        img = transform(img).resize((1024, 1024)).convert('RGB')

        transform_for_original_image = v2.Compose([v2.PILToTensor()])
        img = transform_for_original_image(img)
        
        # calculating the correction needed for the prompt coordinates
        prompt_x_shift = left_x_for_cropping - left_padding_size
        prompt_y_shift = upper_y_for_cropping - upper_padding_size
        prompt = torch.tensor([x_1 - prompt_x_shift, y_1 - prompt_y_shift, x_2 - prompt_x_shift, y_2 - prompt_y_shift])

        # the dictionary is used to know how to crop the annotation outside of the Dataset class
        annotation_shape_adjustment_dict = {}
        annotation_shape_adjustment_dict["upper_y_for_cropping"] = torch.tensor([upper_y_for_cropping])
        annotation_shape_adjustment_dict["lower_y_for_cropping"] = torch.tensor([lower_y_for_cropping])
        annotation_shape_adjustment_dict["left_x_for_cropping"] = torch.tensor([left_x_for_cropping])
        annotation_shape_adjustment_dict["right_x_for_cropping"] = torch.tensor([right_x_for_cropping])
        annotation_shape_adjustment_dict["padding"] = torch.tensor(
            [left_padding_size, right_padding_size, upper_padding_size, lower_padding_size])

        return img, prompt, annotation, annotation_shape_adjustment_dict, annotation_H, annotation_W


    def __getitem__(self, index):
        '''
        Args:
            index (int): the index of the image and annotation in the lists of images and annotations.
        
        return:
            a dictionary of Tensors of different shapes. The dictionary contains the inputs needed to for the forward pass of MedSAM and also for data visualizations.
        '''
        
        # getting the file paths
        img_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]
        
        # creating the transformation from PIL to PyTorch Tensor
        transform_for_original_image = v2.Compose([v2.PILToTensor()])

        original_grayscale_image = Image.open(img_path)
        W, H = original_grayscale_image.size
        original_rgb_image= original_grayscale_image.resize((1024, 1024)).convert('RGB')  # 1024 X 1024 because that's the only possible MedSAM input
        img = self.transforms(original_rgb_image)
        original_annotation = Image.open(annotation_path)
        # annotation = self.transforms(original_annotation.resize((256, 256)))
        annotation = self.transforms(original_annotation)
        img = torch.unsqueeze(img, 0)

        # image_name = img_path[img_path.find('hashed'):].replace('.jpg', '')  # hashed_#_#

        # prepare image and prompt for the model
        if self.YOLO_models is None:  # this never happens now
            inputs = self.processor(img, return_tensors="pt")
        else:
            YOLO_model_index = self.match_yolo_to_image(annotation_path)
            if YOLO_model_index == -1:  # this never happens
                prompt = torch.tensor([-1, -1, -1, -1])
                print(-1111111)
            else:
                YOLO_model = self.YOLO_models[YOLO_model_index]
                YOLO_output = YOLO_model(Image.open(img_path), verbose=False)[0].boxes.xyxy.cpu()
                if torch.numel(YOLO_output) != 0:
                    prompt = YOLO_output[0]  # save only the first (most confident) prediction
                    if self.subimage_input:
                        img, prompt, _, adjustment_dict, annotation_H, annotation_W = self.crop_image_and_prompt(transform_for_original_image(Image.open(img_path)), prompt, transform_for_original_image(annotation), W, H)
                        # note: annotation_W should be equal to annotation_H
                        prompt = prompt / torch.tensor([annotation_W, annotation_H, annotation_W, annotation_H]) * 1024
                    else:
                        prompt = prompt / torch.tensor([W, H, W, H]) * 1024  # make the prompt size fit a 1024X1024 image
                    prompt = prompt.tolist() # make it as a list so it fits the SAM processor
                else:  # if no object was found in the image
                    if self.subimage_input:
                        img, prompt, _, adjustment_dict, annotation_H, annotation_W = self.crop_image_and_prompt(transform_for_original_image(Image.open(img_path)), YOLO_output, transform_for_original_image(annotation), W, H) 
                        prompt = prompt.tolist()
                    else:
                        prompt = [0, 0, 0, 0]
            inputs = self.processor(img, input_boxes=[[prompt]], return_tensors="pt")
        
        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # inputs['pixel_values'] = torch.unsqueeze(inputs['pixel_values'], 0)
        # add ground truth segmentation
        inputs["ground_truth_mask"] = annotation
        original_rgb_image = transform_for_original_image(original_rgb_image)
        inputs["original_rgb_image"] = original_rgb_image
        inputs["prompt"] = torch.tensor(prompt)
        
        if self.subimage_input:
            original_rgb_image = img
        numpy_img = original_rgb_image.numpy().astype(np.uint8)
        numpy_img = (numpy_img - numpy_img.min()) / np.clip(
            numpy_img.max() - numpy_img.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)

        img_1024_tensor = torch.tensor(numpy_img).float()
        inputs["img_1024_tensor"] = img_1024_tensor
        
        # transfer box_np to 1024x1024 scale
        # box_np = [prompt]
        # inputs["box_1024"] = torch.tensor(box_np / np.array([W, H, W, H]) * 1024)
        if self.subimage_input:
            inputs["H"] = torch.tensor([annotation_H])
            inputs["W"] = torch.tensor([annotation_W])
            inputs.update(adjustment_dict)  # merging the two dictionaries
        else:
            inputs["H"] = torch.tensor([H])
            inputs["W"] = torch.tensor([W])
        
        inputs["zero_yolo_detections"] = torch.tensor([torch.numel(YOLO_output) == 0])
        
        return inputs


def train_or_val(epoch_num, medsam_model, dataloader, optimizer, seg_loss, ce_loss, device, batch_size, experiment_dir_path, dice_loss_weight=0.8, train_flag=True, update_model_when_zero_dets=True):
    # train_flag: if it's True, this is training. If False, the model is under validation.
    
    enlargement_factor = dataloader.dataset.dataset.bb_enlargement_factor
    dice_scores = []
    jaccard_scores = []
    hausdorff_distances = []
    hausdorff_distances_without_off_yolo_preds = []
    
    epoch_losses = []
    batch_num = -1
    for batch in tqdm(dataloader):
            batch_num += 1
            running_loss = 0
            # pixel_values = batch['pixel_values']
            box = batch["prompt"]
            original_rgb_image = batch["original_rgb_image"]
            gt = batch["ground_truth_mask"].to(device)
            img_1024_tensor = batch["img_1024_tensor"].to(device)
            box_1024 = batch["prompt"].squeeze()
            
            # if there isn't a model that wasn't trained on this image, move to the next image. This no longer happens because of a change to the dataset folder that I did.
            if torch.sum(box) < 0:
                continue

            for i in range(batch_size):  # the loop is needed because of a problem that occurs while inferring when batch_size > 1
                # cropping the annotation (gt)
                if dataloader.dataset.dataset.subimage_input:
                    upper_y_for_cropping = batch["upper_y_for_cropping"][i]
                    lower_y_for_cropping = batch["lower_y_for_cropping"][i]
                    left_x_for_cropping = batch["left_x_for_cropping"][i]
                    right_x_for_cropping = batch["right_x_for_cropping"][i]
                    padding = tuple(batch["padding"][i].tolist())
                    gt = gt[i, 0, upper_y_for_cropping:lower_y_for_cropping, left_x_for_cropping:right_x_for_cropping] # cropping
                    gt = nn.functional.pad(gt, padding) # in most cases this line doesn't do anything
                
                H = batch["H"][i].item()
                W = batch["W"][i].item()
                
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(img_1024_tensor[i].unsqueeze(0)) # the output size is (1, 256, 64, 64)
                prompt = batch["prompt"][i].unsqueeze(0)
                seg, medsam_pred = medsam_inference(medsam_model, image_embedding, prompt, H=H, W=W, tensor_output=True)  # 385X627 because of the shape of gt
                gt = gt.squeeze()
                medsam_pred = medsam_pred.to(device)
                loss = dice_loss_weight * seg_loss(medsam_pred, gt) + (1 - dice_loss_weight) * ce_loss(medsam_pred, gt.float())
                if train_flag:
                    if update_model_when_zero_dets or (not update_model_when_zero_dets and not batch["zero_yolo_detections"][i]):
                        loss.backward()
                        optimizer.step()

                optimizer.zero_grad()
                
                annotation = gt.cpu().numpy().squeeze()
                seg = seg.cpu().numpy().squeeze()
                boolean_gt = annotation != 0
                boolean_pred = seg != 0
                intersection = np.logical_and(boolean_gt, boolean_pred).sum()
                union = np.logical_or(boolean_gt, boolean_pred).sum()
                dice = 2 * intersection / (np.sum(boolean_gt) + np.sum(boolean_pred) + 0.0000001)
                jaccard = intersection / (np.sum(boolean_gt) + np.sum(boolean_pred) - intersection + 0.0000001)
                hausdorff_distance_score = hausdorff_distance(boolean_gt, boolean_pred)
                
                dice_scores += [dice]
                jaccard_scores += [jaccard]
                # if YOLO finds an object that has no overlap with the ground truth, the cropping of the ground truth image results in an image with only zeros and the calculated HD is infinity.
                # the solution is either ignoring the infinity or padding the prediction with 
                if not math.isinf(hausdorff_distance_score):
                    hausdorff_distances += [hausdorff_distance_score]
                    hausdorff_distances_without_off_yolo_preds += [hausdorff_distance_score]
                else:
                    # putting the annotation and the prediction in an image with the size of the original image
                    large_boolean_pred = np.zeros_like(batch["ground_truth_mask"].to(device).cpu().numpy().squeeze())
                    large_boolean_pred[upper_y_for_cropping:lower_y_for_cropping, left_x_for_cropping:right_x_for_cropping] = boolean_pred
                    large_boolean_gt = batch["ground_truth_mask"].to(device).cpu().numpy().squeeze() > 0 
                    hausdorff_distance_score = hausdorff_distance(large_boolean_gt, large_boolean_pred)
                    hausdorff_distances += [hausdorff_distance_score]
                
                # plotting only some images and not in every epoch. Doing it only when validating
                if not train_flag and batch_num < 15 and (epoch_num % 25 == 0 or epoch_num == -1):
                    if dataloader.dataset.dataset.bb_enlargement_factor > 1:
                        medsam_pred = draw_bb_on_pred(medsam_pred, box_1024, shrink_factor=1024 / H)
                    
                    pred_compared_to_gt = visualize_segmentation(boolean_gt, boolean_pred)
                    plot0 = plt.imshow(pred_compared_to_gt.astype('uint8'))
                    plt.title('Dice: ' + str(dice)[:5])
                    plt.savefig(experiment_dir_path + '/validation_img_number_' + str(batch_num) + '_epoch_num_' + str(epoch_num) + '_pred-GT_comparison.png')
                    plt.clf() 
                    
                    plot1 = plt.imshow(medsam_pred.cpu().detach().numpy())
                    plt.title('Dice: ' + str(dice)[:5])
                    plt.savefig(experiment_dir_path + '/validation_img_number_' + str(batch_num) + '_epoch_num_' + str(epoch_num) + '_pred.png')
                    plt.clf()
                    # plotting the original image
                    if epoch_num == 0:
                        plot4 = plt.imshow(img_1024_tensor[i].permute((1,2,0)).squeeze().cpu().numpy(), cmap='gray')
                        plt.savefig(experiment_dir_path + '/validation_img_number_' + str(batch_num) + '_epoch_num_' + str(epoch_num) + '_model_image_input.png')
                        plt.clf()
                        rescaled = torch.nn.functional.interpolate(original_rgb_image, size=(385, 627), mode='bilinear').squeeze().permute((1,2,0)).cpu().detach().numpy()
                        plot2 = plt.imshow(rescaled, cmap='gray')
                        plt.savefig(experiment_dir_path + '/validation_img_number_' + str(batch_num) + '_epoch_num_' + str(epoch_num) + '_original.png')
                        plt.clf()
                        plot3 = plt.imshow(gt.squeeze().cpu().detach().numpy() * 255)
                        plt.savefig(experiment_dir_path + '/validation_img_number_' + str(batch_num) + '_epoch_num_' + str(epoch_num) + '_ground_truth.png')
                        plt.clf()
                    
                epoch_losses.append(loss.item())
    
    if train_flag:
        return epoch_losses
    else:
        return epoch_losses, dice_scores, jaccard_scores, hausdorff_distances, hausdorff_distances_without_off_yolo_preds


# the code in this file is partly based on the next :
# https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
def main(update_model_when_zero_dets):
    dice_loss_weight = 0.5
    load_previous_dls = True  # used to compare one training session more easily to other training sessions that had the same dataloader
    data_split = [0.75, 0.20, 0.05]  # train, val, test
    batch_size = 1  # the only possible batch size when I tried doing the forward pass of MedSAM. Other batch sizes give an error
    num_workers = 4
    num_epochs = 150  # TODO change to 400 when the training setup is decided
    training_patience = 100

    MedSAM_CKPT_PATH = "/workspace/PycharmProjects/pythonProject/medsam_vit_b.pth"
    device = "cuda:0"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)

    # make sure we only compute gradients for mask decoder
    # for name, param in medsam_model.named_parameters():
    #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #         param.requires_grad_(False)

    # Initialize the optimizer and the loss function
    # optimizer = Adam(medsam_model.mask_decoder.parameters(), lr=0.00005, weight_decay=0.0005)  # originally the WD was 0 and LR was 1e-4
    optimizer = torch.optim.SGD(medsam_model.mask_decoder.parameters(), lr=0.005*0.1, momentum=0.9, weight_decay=0.0005)
    # Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(include_background=False, sigmoid=True, squared_pred=True, reduction='mean')
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # the weights of the YOLOv8 models that are used to create the bounding box prompt for MedSAM
    YOLO_path = "/workspace/runs/detect/semi-supervised_XL_model_training-series1_32-fold_increase_43904/weights/best.pt"
    YOLO_path2 = "/workspace/runs/detect/semi-supervised_XL_model_training-series2_32-fold_increase_43831/weights/best.pt"
    YOLO_path3 = "/workspace/runs/detect/semi-supervised_XL_model_training-series3_32-fold_increase_44119/weights/best.pt"
    YOLO_paths = [YOLO_path, YOLO_path2, YOLO_path3]
    images_dir = '/workspace/PycharmProjects/pythonProject/datasets/images'
    annotations_dir = '/workspace/PycharmProjects/pythonProject/datasets/png segmentations'
    
    # loading the YOLOv8 weights
    YOLO_models = [YOLO(YOLO_path) for YOLO_path in YOLO_paths]
    
    if load_previous_dls:  # if you would like to load previous dataloader and not create new ones with a new random split, do this
        val_loader, test_loader, train_loader = utils.load_dls(experiment_dir='/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/20240715_152223_dice_loss_weight_0.5')
        train_ds_size = len(train_loader.dataset.dataset)
        val_ds_size = len(val_loader.dataset.dataset)
        test_ds_size = len(test_loader.dataset.dataset)
        full_ds_size = train_ds_size + val_ds_size + test_ds_size
    else:
        full_ds = SAM_dataset(images_dir, annotations_dir, processor, YOLO_models=YOLO_models, transforms=None, subimage_input=True)
        full_ds_size = len(full_ds)
        generator = torch.Generator().manual_seed(42)  # Generator used for the random permutation
        indices, dataset_sizes, _ = utils.find_best_data_split(images_path=annotations_dir,
                                                                                train_share=data_split[0],
                                                                                val_share=data_split[1],
                                                                                test_share=data_split[2],
                                                                                segmentation_task=True)
        train_ds, val_ds, test_ds = utils.random_split(indices, dataset_sizes, full_ds, data_split, generator=generator)
    
    
        train_ds_size = len(train_ds)
        val_ds_size = len(val_ds)
        test_ds_size = len(test_ds)
    
        # creating the random sampler
        # sex_array = np.load('sexlist_without_missing_patients.npy')
        # train_sex_array = sex_array[indices[:train_ds_size]]  # getting only the sexes of tagged training videos
        # train_sampler = utils.MyRandomWeightedSampler(data_source=train_ds, sex_mask=train_sex_array,
        #                                               replacement=False)
        
        # creating the dataloaders
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True,
                                  pin_memory=True
                                  )
        val_loader = DataLoader(val_ds,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=False,
                              pin_memory=True
                              )
        test_loader = DataLoader(test_ds,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=False,
                              pin_memory=True
                              )

    print("Created dataloaders.")
    print('The whole dataset has {} instances'.format(full_ds_size))
    print('The training dataset has {} instances'.format(train_ds_size))
    print('The validation dataset has {} instances'.format(val_ds_size))
    print('The test dataset has {} instances'.format(test_ds_size))

    best_dice = 0
    best_jaccard = 0
    best_hausdorff = 1000000 # an arbitrarily large number
    early_stopping_counter = 0

    # these are used to create graphs of the model performance per epoch
    dice_per_epoch = []
    jaccard_per_epoch = []
    hausdorff_per_epoch = []
    hausdorff_without_off_yolo_per_epoch = []
    train_losses = []
    val_losses = []
    
    experiment_dir_path = '/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_dice_loss_weight_' + str(dice_loss_weight)
    os.mkdir(experiment_dir_path)  # creating an experiment folder
    # utils.save_dls(experiment_dir_path, val_loader, test_loader, train_loader)
    utils.log_variables(experiment_dir_path, experiment_name='Checking optimizer', device=device, optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs, num_workers=num_workers, data_split=data_split, max_epochs_without_improvement=training_patience, train_ds_size=train_ds_size, val_ds_size=val_ds_size, test_ds_size=test_ds_size)
    # the next three lines are used to show the performance of MedSAM in zero-shot segmentation on the validation set
    epoch = -1
    medsam_model.eval()

    val_loss, dice_scores, jaccard_scores, hausdorff_distances, hausdorff_distances_without_off_yolo_preds = train_or_val(epoch, medsam_model, val_loader, optimizer, seg_loss, ce_loss, device, batch_size, experiment_dir_path, dice_loss_weight=dice_loss_weight, train_flag=False)
    
    for epoch in range(num_epochs):
        medsam_model.train()  # put in train mode
        train_loss = train_or_val(epoch, medsam_model, train_loader, optimizer, seg_loss, ce_loss, device, batch_size, experiment_dir_path, dice_loss_weight=dice_loss_weight, train_flag=True, update_model_when_zero_dets=update_model_when_zero_dets)
        train_losses += [mean(train_loss)]
        
        print(f'EPOCH: {epoch}')
        print(f'Mean train loss: {mean(train_loss)}')
        
        medsam_model.eval()  # put in validation mode
        val_loss, dice_scores, jaccard_scores, hausdorff_distances, hausdorff_distances_without_off_yolo_preds = train_or_val(epoch, medsam_model, val_loader, optimizer, seg_loss, ce_loss, device, batch_size, experiment_dir_path, dice_loss_weight=dice_loss_weight, train_flag=False)
        val_losses += [mean(val_loss)]
        print(f'Mean val loss: {mean(val_loss)}')
        #  inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # calculating metrics for the epoch
        mean_epoch_dice = np.mean(np.array(dice_scores))
        mean_epoch_jaccard = np.mean(np.array(jaccard_scores))
        mean_epoch_hausdorff = np.mean(np.array(hausdorff_distances))
        mean_epoch_hausdorff_without_off_yolo_preds = np.mean(np.array(hausdorff_distances_without_off_yolo_preds))
        
        # saving in variables the metrics of this epoch
        dice_per_epoch += [mean_epoch_dice]
        jaccard_per_epoch += [mean_epoch_jaccard]
        hausdorff_per_epoch += [mean_epoch_hausdorff]
        hausdorff_without_off_yolo_per_epoch += [mean_epoch_hausdorff_without_off_yolo_preds]
        print('dices:')
        print(dice_per_epoch)
        # if there's an improvement in the dice similarity score, save it
        if mean_epoch_dice > best_dice:
            early_stopping_counter = 0
            best_dice = mean_epoch_dice
            best_jaccard = mean_epoch_jaccard
            best_hausdorff = mean_epoch_hausdorff
        else:
            early_stopping_counter += 1  # counting the number of epochs without improvement in dice score

        # if too many epochs have passed without an improvement in the dice score or if it's the last epoch
        if early_stopping_counter >= training_patience or epoch == num_epochs - 1:
            print('best dice: ' + str(best_dice) + '. best jaccarad: ' + str(best_jaccard))
            print('best Hausdorff distance: ' + str(best_hausdorff))
            print('best Jaccard distance: ' + str(best_jaccard))

            best_results = {'dice': best_dice,
                       'Hausdorff': best_hausdorff,
                            'Jaccard': best_jaccard}
            string_dice = str(best_dice)[:5]

            with open(experiment_dir_path + "/dice_" + string_dice + ".json", "w") as outfile:
                json.dump(best_results, outfile)
            
            plt.clf()
            plt.plot(dice_per_epoch, label='Dice Score')
            plt.plot(jaccard_per_epoch, label='Jaccard Score')
            plt.legend(loc="upper left")
            plt.xlabel('Epoch number')
            plt.savefig(experiment_dir_path + '/performance_per_epoch' + '.png')
            plt.clf()  # clearing the plot for the next plot

            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend(loc="upper left")
            plt.xlabel('Epoch number')
            plt.savefig(experiment_dir_path + '/losses_per_epoch' + '.png')
            plt.clf()  # clearing the plot for the next plot
            
            plt.plot(hausdorff_per_epoch, label='Hausdorff Distance')
            plt.legend(loc="upper left")
            plt.xlabel('Epoch number')
            plt.savefig(experiment_dir_path + '/hausdorff_distance_per_epoch' + '.png')
            plt.clf()  # clearing the plot for the next plot
            
            plt.plot(hausdorff_without_off_yolo_per_epoch, label='Hausdorff Distance')
            plt.legend(loc="upper left")
            plt.xlabel('Epoch number')
            plt.savefig(experiment_dir_path + '/hausdorff_distance_per_epoch_without_off_yolo_preds' + '.png')
            plt.clf()  # clearing the plot for the next plot
            
            # save the performance metrics in a CSV file
            with open(experiment_dir_path + '/performance_per_epoch.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['train_losses', 'val_losses', 'dice_per_epoch', 'jaccard_per_epoch', 'hausdorff_per_epoch', 'hausdorff_per_epoch_without_off_yolo_preds']) # Write the column names
                
                # Write the data rows by zipping the lists
                for row in zip(train_losses, val_losses, dice_per_epoch, jaccard_per_epoch, hausdorff_per_epoch, hausdorff_without_off_yolo_per_epoch):
                  writer.writerow(row)

            break

# TODO change the random sampler

if __name__ == '__main__':
    # folder_path = '/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/20240706_205925_dice_loss_weight_0.4'
    # for filename in os.listdir(folder_path):
    #     if "original" in filename and "num_0" not in filename:
    #       file_path = os.path.join(folder_path, filename)
    #       os.remove(file_path)
          
    update_model_when_zero_dets_options = [True]
    for update_model_when_zero_dets_option in update_model_when_zero_dets_options:
        main(update_model_when_zero_dets_option)
