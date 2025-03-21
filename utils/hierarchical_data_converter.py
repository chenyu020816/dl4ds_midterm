import argparse
import os
import shutil
from tqdm import tqdm

from .utils import coarse_to_fine 


def create_coarse_data(src_path, dst_path, mapping):
    os.makedirs(dst_path, exist_ok=True)
    
    for coarse_label, fine_labels in tqdm(mapping.items(), desc="Creating Coarse Dataset"):
        coarse_folder = os.path.join(dst_path, coarse_label)
        os.makedirs(coarse_folder, exist_ok=True)

        for fine_label in fine_labels:
            fine_folder = os.path.join(src_path, coarse_label, fine_label)
            if not os.path.exists(fine_folder):
                continue

            for img in os.listdir(fine_folder):
                src_img_path = os.path.join(fine_folder, img)
                dst_img_path = os.path.join(coarse_folder, img)
                shutil.copy(src_img_path, dst_img_path)


def create_fine_data(src_path, dst_path):
    shutil.copytree(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, default='./data/aug_noise_data_v2_5')
    parser.add_argument('--output_folder', type=str, default='./data/hierarchical_data')
    args = parser.parse_args()

    coarse_folder = os.path.join(args.output_folder, "coarse_data")
    fine_folder = os.path.join(args.output_folder, "fine_data")
    create_coarse_data(args.source_data, coarse_folder, coarse_to_fine)
    create_fine_data(args.source_data, fine_folder)

    