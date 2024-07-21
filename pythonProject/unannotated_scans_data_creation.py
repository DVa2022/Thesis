from utils import dataset_preparation_for_unannotated_scans


def main():
    dataset_preparation_for_unannotated_scans(skip_raw_frames_creation=True)


if __name__ == '__main__':
    main()
