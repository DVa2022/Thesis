from utils import multiple_inference_videos_creation
import os


def main():

    vid_names = os.listdir('/workspace/PycharmProjects/pythonProject/datasets/valid/labels')
    vid_names = ['_'.join(vid.split('_')[3:5]) for vid in vid_names]
    model_dir = 'semi-supervised_XL_model_training-series1_32-fold_increase_43904'
    multiple_inference_videos_creation(model_dir=model_dir, vid_names=vid_names[:40])


if __name__ == '__main__':
    main()
