import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from ultralytics import YOLO
from transformers import SamProcessor
import torch.nn as nn

import pdb
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

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def medsam_inference(medsam_model, img_embed, box_1024, H, W, tensor_output=False):
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

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu()
    if tensor_output:
        medsam_seg = (low_res_pred > 0.5).to(torch.int32)
    else:
        low_res_pred = low_res_pred.numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    
    return medsam_seg, low_res_pred


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

        self.image_paths = [os.path.join(images_dir, short_to_long_img_name_conversion_dict[path]) for path in
                            annotation_list]
        self.annotations_dir = annotations_dir
        self.annotation_paths = [os.path.join(annotations_dir, path) for path in annotation_list]

        self.YOLO_models = YOLO_models
        self.processor = processor

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

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
        # this function was used only once
        
        import shutil
        
        for index in range(10000):
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


    def __getitem__(self, index):
        img_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]

        transform_for_original_image = v2.Compose([v2.PILToTensor()])

        original_grayscale_image = Image.open(img_path)
        W, H = original_grayscale_image.size
        original_grayscale_image= original_grayscale_image.resize((1024, 1024)).convert('RGB')
        img = self.transforms(original_grayscale_image)
        annotation = self.transforms(Image.open(annotation_path))  # .resize((256, 256)))
        img = torch.unsqueeze(img, 0)

        # image_name = img_path[img_path.find('hashed'):].replace('.jpg', '')  # hashed_#_#

        # prepare image and prompt for the model
        if self.YOLO_models is None:
            inputs = self.processor(img, return_tensors="pt")
        else:
            YOLO_model_index = self.match_yolo_to_image(annotation_path)
            if YOLO_model_index == -1:
                prompt = torch.tensor([-1, -1, -1, -1])
                print(-1111111)
            else:
                YOLO_model = self.YOLO_models[YOLO_model_index]
                YOLO_output = YOLO_model(Image.open(img_path))[0].boxes.xyxy.cpu()
                if torch.numel(YOLO_output) != 0:
                    prompt = YOLO_output[0]  # save only the first (most confident) prediction
                    prompt = prompt / torch.tensor([W, H, W, H]) * 1024 # make the prompt size fit a 1024X1024 image
                    prompt = prompt.tolist() # make it as a list so it fits the SAM processor
                else:  # if no object was found in the image
                    prompt = [0, 0, 0, 0]
            inputs = self.processor(img, input_boxes=[[prompt]], return_tensors="pt")
        
        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['pixel_values'] = torch.unsqueeze(inputs['pixel_values'], 0)
        # add ground truth segmentation
        inputs["ground_truth_mask"] = annotation
        original_grayscale_image = transform_for_original_image(original_grayscale_image)
        inputs["original_grayscale_image"] = transform_for_original_image(original_grayscale_image)
        inputs["prompt"] = torch.tensor(prompt) # rescale the prompt to the size of the input image
        
        numpy_img = original_grayscale_image.numpy().astype(np.uint8)
        numpy_img = (numpy_img - numpy_img.min()) / np.clip(
            numpy_img.max() - numpy_img.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)

        img_1024_tensor = torch.tensor(numpy_img).float()
        inputs["img_1024_tensor"] = img_1024_tensor
        
        # transfer box_np to 1024x1024 scale
        # box_np = [prompt]
        # inputs["box_1024"] = torch.tensor(box_np / np.array([W, H, W, H]) * 1024)
        inputs["H"] = torch.tensor([H])
        inputs["W"] = torch.tensor([W])

        return inputs


def train_or_val(medsam_model, dataloader, optimizer, seg_loss, ce_loss, device, batch_size, train_flag=True):
    # train_flag: if it's True, this is training. If False, the model is under validation.
    
    if not train_flag:
        dice_scores = []
        jaccard_scores = []
        hausdorff_distances = []
    
    epoch_losses = []
    # j = -1
    for batch in tqdm(dataloader):
            # j += 1
            running_loss = 0
            pixel_values = batch['pixel_values']
            box = batch["prompt"]
            test_image = batch["original_grayscale_image"]
            gt = batch["ground_truth_mask"].to(device)
            img_1024_tensor = batch["img_1024_tensor"].to(device)
            box_1024 = batch["prompt"].squeeze()
            
            H = batch["H"][0].item()
            W = batch["W"][0].item()
            
            # if there isn't a model that wasn't trained on this image, move to the next image. This no longer happens because of a change to the dataset folder that I did.
            if torch.sum(box) < 0:
                continue

            for i in range(batch_size):  # the loop is needed because of a problem that occurs while inferring when batch_size > 1
                # pdb.set_trace()
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(img_1024_tensor[i].unsqueeze(0)) # the output size is (1, 256, 64, 64)
                prompt = batch["prompt"][i].unsqueeze(0)
                seg, medsam_pred = medsam_inference(medsam_model, image_embedding, prompt, H=385, W=627, tensor_output=True)  # 385X627 because of the shape of gt
                gt = gt.squeeze()
                medsam_pred = medsam_pred.to(device)
                loss = seg_loss(medsam_pred, gt) + ce_loss(medsam_pred, gt.float())
                if train_flag:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                
                # prompt = box_1024 / 4
                # x1 = int(prompt[0].item())
                # x2 = int(prompt[2].item())
                # y1 = int(prompt[1].item())
                # y2 = int(prompt[3].item())
                # medsam_pred = medsam_pred.squeeze().cpu()
                # pdb.set_trace()
                # medsam_pred[y1:y2, x1] = 0.5
                # medsam_pred[y1:y2, x2] = 0.5
                # medsam_pred[y1, x1:x2] = 0.5
                # medsam_pred[y2, x1:x2] = 0.5
                # plot1 = plt.imshow(medsam_pred.cpu().detach().numpy())
                # plt.savefig('/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/' + str(j) + '_pred.png')
                # plot2 = plt.imshow(test_image.squeeze().cpu().detach().numpy()[0])
                # plt.savefig('/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/' + str(j) + '.png')
                # plt.clf()
                if not train_flag:  # calculate segmentation metrics when validating
                    pdb.set_trace()
                    annotation = transform.resize(gt[i].cpu().numpy().squeeze(), (640, 640))
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
                    
                epoch_losses.append(loss.item())
    # pdb.set_trace()
    if train_flag:
        return epoch_losses
    else:
        return epoch_losses, dice_scores, jaccard_scores, hausdorff_distances


# the following implementation is based on:
# https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
def main():
    data_split = [0.75, 0.20, 0.05]  # train, val, test
    batch_size = 1
    num_workers = 4
    num_epochs = 5

    MedSAM_CKPT_PATH = "/workspace/PycharmProjects/pythonProject/medsam_vit_b.pth"
    device = "cuda:0"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)

    # make sure we only compute gradients for mask decoder
    for name, param in medsam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Initialize the optimizer and the loss function
    optimizer = Adam(medsam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    # Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    YOLO_path = "/workspace/runs/detect/semi-supervised_XL_model_training-series1_32-fold_increase_43904/weights/best.pt"
    YOLO_path2 = "/workspace/runs/detect/semi-supervised_XL_model_training-series2_32-fold_increase_43831/weights/best.pt"
    YOLO_path3 = "/workspace/runs/detect/semi-supervised_XL_model_training-series3_32-fold_increase_44119/weights/best.pt"
    YOLO_paths = [YOLO_path, YOLO_path2, YOLO_path3]
    images_dir = '/workspace/PycharmProjects/pythonProject/datasets/images'
    annotations_dir = '/workspace/PycharmProjects/pythonProject/datasets/png segmentations'

    YOLO_models = [YOLO(YOLO_path) for YOLO_path in YOLO_paths]
    full_ds = SAM_dataset(images_dir, annotations_dir, processor, YOLO_models=YOLO_models, transforms=None)

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

    # these two are used to create the graph of the model performance per epoch
    dice_per_epoch = []
    jaccard_per_epoch = []

    for epoch in range(num_epochs):
        medsam_model.train()
        # train_loss = train_or_val(medsam_model, train_loader, optimizer, seg_loss, ce_loss, device, batch_size, train_flag=True)
        
        # print(f'EPOCH: {epoch}')
        # print(f'Mean train loss: {mean(train_loss)}')
        
        medsam_model.eval()
        val_loss, dice_scores, jaccard_scores, hausdorff_distances = train_or_val(medsam_model, train_loader, optimizer, seg_loss, ce_loss, device, batch_size, train_flag=False)
        print(f'Mean val loss: {mean(val_loss)}')
        #  inputs = {k: v.to(device) for k, v in inputs.items()}

        mean_epoch_dice = np.mean(np.array(dice_scores))
        mean_epoch_jaccard = np.mean(np.array(jaccard_scores))
        mean_epoch_hausdorff = np.mean(np.array(hausdorff_distances))

        dice_per_epoch += [mean_epoch_dice]
        jaccard_per_epoch += [mean_epoch_jaccard]

        if mean_epoch_dice > best_dice:
            early_stopping_counter = 0
            best_dice = mean_epoch_dice
            best_jaccard = mean_epoch_jaccard
            best_hausdorff = mean_epoch_hausdorff
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 10:
            print('best dice: ' + str(best_dice) + '. best jaccarad: ' + str(best_jaccard))
            print('best Hausdorff distance: ' + str(best_hausdorff))
            print('best Jaccard distance: ' + str(best_jaccard))

            best_results = {'dice': best_dice,
                       'Hausdorff': best_hausdorff,
                            'Jaccard': best_jaccard}
            string_dice = str(best_dice)[:5]

            with open("/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/best_results_" + datetime.now().strftime('%Y%m%d_%H%M%S') + "_dice_" + string_dice + ".json", "w") as outfile:
                json.dump(best_results, outfile)

            plt.plot(dice_per_epoch, label='Dice Score')
            plt.plot(jaccard_per_epoch, label='Jaccard Score')
            plt.legend(loc="upper left")
            plt.xlabel('Epoch number')
            plt.savefig('/workspace/PycharmProjects/pythonProject/datasets/runs of medsam finetuned on our data/performance_per_epoch' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')
            plt.clf()  # clearing the plot for the next plot

            break

# TODO change the random sampler

if __name__ == '__main__':
    main()
