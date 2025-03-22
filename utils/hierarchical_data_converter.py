import argparse
import os
import shutil
from tqdm import tqdm

from utils import coarse_to_fine, class_names, fine_to_coarse


def create_coarse_data(src_path, dst_path):
    coarse_folder = os.path.join(dst_path, "coarse_data")
    fine_folder = os.path.join(dst_path, "fine_data")

    for coarse in coarse_to_fine.keys():
        coarse_class_folder = os.path.join(coarse_folder, coarse)
        os.makedirs(coarse_class_folder, exist_ok=True)
        coarse_fine_classes = coarse_to_fine[coarse]
        for fine_class in coarse_fine_classes:
            fine_class_folder = os.path.join(fine_folder, coarse, fine_class)
            os.makedirs(fine_class_folder, exist_ok=True)

    for cls in tqdm(class_names):
        cls_to_coarse = fine_to_coarse[cls]

        src_image_folder = os.path.join(src_path, cls)
        for img in os.listdir(src_image_folder):
            img_src = os.path.join(src_image_folder, img)
            img_coarse_dst = os.path.join(coarse_folder, cls_to_coarse, img)
            img_fine_dst = os.path.join(fine_folder, cls_to_coarse, cls, img)
            shutil.copy(img_src, img_coarse_dst)
            shutil.copy(img_src, img_fine_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, default='./data/aug_noise_data_v2_5')
    parser.add_argument('--output_folder', type=str, default='./data/hierarchical_data')
    args = parser.parse_args()

    create_coarse_data(args.source_data, args.output_folder)

    