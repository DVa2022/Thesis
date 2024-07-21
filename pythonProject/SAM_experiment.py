import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torchvision
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import math
import sys
import time
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel
from transformers import SamProcessor
from skimage.metrics import hausdorff_distance

# Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class SAM_dataset(Dataset):
    def __init__(self, images_dir, annotations_dir, processor, YOLO_models=None, transforms=None):
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
            
        self.image_paths = [os.path.join(images_dir, short_to_long_img_name_conversion_dict[path]) for path in annotation_list]
        self.annotations_dir = annotations_dir
        self.annotation_paths = [os.path.join(annotations_dir, path) for path in annotation_list]

        self.YOLO_models = YOLO_models
        self.processor = processor
        
        if transforms is not None:
            self.transforms_from_user = transforms
        else:
            self.transforms_from_user = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        
        self.length = len(os.listdir(annotations_dir))
        
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
    
        file_name_in_short_format = "_".join(annotation_path[annotation_path.find('hashed'):].split('_')[:2])  # in the format of 'hashed_#'
        
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

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]
        
        # img = np.asarray(Image.open(img_path))
        # annotation = np.asarray(Image.open(annotation_path))

        original_grayscale_image = Image.open(img_path).resize((640, 640))
        img = self.transforms_from_user(original_grayscale_image.convert('RGB'))
        annotation = self.transforms_from_user(Image.open(annotation_path).resize((256, 256)))
        img = torch.unsqueeze(img, 0)

        # image_name = img_path[img_path.find('hashed'):].replace('.jpg', '')  # hashed_#_#

        # prepare image and prompt for the model
        if self.YOLO_models is None:
            inputs = self.processor(img, return_tensors="pt")
        else:
            YOLO_model_index = self.match_yolo_to_image(annotation_path)
            if YOLO_model_index == -1:
                return -1, -1, -1
            print(YOLO_model_index, YOLO_model_index, YOLO_model_index, YOLO_model_index, YOLO_model_index, YOLO_model_index, YOLO_model_index)
            YOLO_model = self.YOLO_models[YOLO_model_index]
            YOLO_output = YOLO_model(img)[0].boxes.xyxy.cpu()
            if torch.numel(YOLO_output) != 0:
                prompt = YOLO_output.tolist()[0]
            else:
                prompt = [0, 0, 0, 0]
            inputs = self.processor(img, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['pixel_values'] = torch.unsqueeze(inputs['pixel_values'], 0)
        # add ground truth segmentation
        inputs["ground_truth_mask"] = annotation

        return inputs, original_grayscale_image, prompt


# the following implementation is based on:
# https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
def main():

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.to(device)

    if 'stu16' in os.getcwd():
        YOLO_path = "/home/stu16/PycharmProjects/pythonProject/runs/detect/fourth_take_no_dfl_extra_large_model20/weights/best.pt"
        images_dir = "/home/stu16/PycharmProjects/pythonProject/datasets/images"
        annotations_dir = "/home/stu16/PycharmProjects/pythonProject/datasets/png segmentations"
    else:  # if we're on the DGX server
        YOLO_path = "/workspace/runs/detect/semi-supervised_XL_model_training-series1_32-fold_increase_43904/weights/best.pt"
        YOLO_path2 = "/workspace/runs/detect/semi-supervised_XL_model_training-series2_32-fold_increase_43831/weights/best.pt"
        YOLO_path3 = "/workspace/runs/detect/semi-supervised_XL_model_training-series3_32-fold_increase_44119/weights/best.pt"
        YOLO_paths = [YOLO_path, YOLO_path2, YOLO_path3]
        images_dir = '/workspace/PycharmProjects/pythonProject/datasets/images'
        annotations_dir = '/workspace/PycharmProjects/pythonProject/datasets/png segmentations'

    YOLO_models = [YOLO(YOLO_path) for YOLO_path in YOLO_paths]
    dataset = SAM_dataset(images_dir, annotations_dir, processor, YOLO_models=YOLO_models, transforms=None)
    
    dice_scores = []
    jaccard_scores = []
    hausdorff_distances = []
    print('checkkkkkkkkkk')
    for i in range(len(dataset)):
        inputs, test_image, box = dataset[i]
        if inputs == -1:  # if there isn't a model that wasn't trained on this image, move to the next image
            continue
        # Move the input tensor to the GPU if it's not already there
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            inputs['input_boxes'] = torch.unsqueeze(inputs['input_boxes'], 1)
            outputs = model(pixel_values=inputs['pixel_values'], input_boxes=inputs['input_boxes'], ground_truth_mask=inputs['ground_truth_mask'], multimask_output=False)
            # outputs = model(**inputs, multimask_output=False)
    
        annotation = inputs["ground_truth_mask"].cpu().numpy().squeeze()
    
        # apply sigmoid
        seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        seg_prob = seg_prob.cpu().numpy().squeeze()
        seg = (seg_prob > 0.5).astype(np.uint8)
    
        boolean_gt = annotation != 0
        boolean_pred = seg != 0
        intersection = np.sum(np.logical_and(boolean_gt, boolean_pred))
        union = np.sum(np.logical_or(boolean_gt, boolean_pred))
        dice = 2 * intersection / (np.sum(boolean_gt) + np.sum(boolean_pred) + 0.0000001)
        jaccard = intersection / (np.sum(boolean_gt) + np.sum(boolean_pred) - intersection + 0.0000001)
        hausdorff_distance_score = hausdorff_distance(boolean_gt, boolean_pred)
        dice_scores += [dice]
        jaccard_scores += [jaccard]
        hausdorff_distances += [hausdorff_distance_score]
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # im = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # images_w_rectangle += [im]
        # cv2.rectangle(im, (int(box[0, 0].item()), int(box[0, 1].item())),
        #               (int(box[0, 2].item()), int(box[0, 3].item())),
        #               color=(0, 0, 255), thickness=1)  # red color
        
      
        axes[0].imshow(test_image, cmap='gray')
        axes[0].set_title("Image")

        axes[1].imshow(seg_prob)
        axes[1].set_title("Probability Map")
      
        axes[2].imshow(seg, cmap='gray')
        axes[2].set_title("Predicted mask")
      
        axes[3].imshow(annotation, cmap='gray')
        axes[3].set_title("Ground truth. Dice: " + str(dice)[:5] + ". Jaccard: " + str(jaccard)[:5])
      
        # Hide axis ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
      
        # Display the images side by side
        # plt.show()
      
        plt.savefig('/workspace/PycharmProjects/pythonProject/datasets/segmentation results/segmentation result_' + str(i) + '.png')
        
        # to save memory, close the figure
        plt.close()
    
    print('average dice: ' + str(np.mean(np.array(dice_scores))) + '. average jaccarad: ' + str(np.mean(np.array(jaccard_scores))))
    print('average Hausdorff distance: ' + str(np.mean(np.array(hausdorff_distances))))
    print('STD of dice: ' + str(np.std(np.array(dice_scores))) + '. STD jaccarad: ' + str(np.std(np.array(jaccard_scores))))
    print('STD of Hausdorff distance: ' + str(np.std(np.array(hausdorff_distances))))
    np.savetxt('/workspace/PycharmProjects/pythonProject/datasets/sam_results_dice.csv', np.array(dice_scores), delimiter=",")
    np.savetxt('/workspace/PycharmProjects/pythonProject/datasets/sam_results_jaccarrd.csv', np.array(jaccard_scores), delimiter=",")
    np.savetxt('/workspace/PycharmProjects/pythonProject/datasets/sam_results_hausdorff.csv', np.array(hausdorff_distances), delimiter=",")


if __name__ == '__main__':
    main()
