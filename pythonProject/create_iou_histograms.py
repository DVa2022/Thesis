from yolo_experiment import get_val_results, yolo_to_pascal_voc
from ultralytics import YOLO
import os

# this file was created because originally when I trained the YOLOv8 models with the training sets that were created
# using semi-supervised learning, I didn't create after each training session a histogram of the IoUs. In this file
# I do it for some of the so-far best models.


def main():

    models = ['semi-supervised_XL_model_training-series1_32-fold_increase_43904',
              'semi-supervised_XL_model_training-series2_32-fold_increase_43831',
              'semi-supervised_XL_model_training-series3_32-fold_increase_44119',
              'semi-supervised_XL_model_training-series1_52-fold_increase_71770',
              'semi-supervised_XL_model_training-series2_52-fold_increase_71600_2',
              'semi-supervised_XL_model_training-series3_52-fold_increase_72072'
              ]
    inference_options = ['valid', 'test']
    models_path = '/workspace/runs/detect/'

    for model in models:
        model_weights = YOLO(models_path + model + '/weights/best.pt')
        for inference_option in inference_options:
            get_val_results(model_weights, saving_dir=model, valid_or_test=inference_option)


if __name__ == '__main__':
    main()
