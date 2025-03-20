import argparse
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


def analysis_cm(cm_path, top_n):
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm_df.index = class_names
    cm_df.columns = class_names
    error_data = []
    class_totals = cm_df.sum(axis=1)

    for true_class in class_names:
        for pred_class in class_names:
            if true_class != pred_class and cm_df.loc[true_class, pred_class] > 0:
                prop = cm_df.loc[true_class, pred_class] / class_totals[true_class]
                error_data.append({
                    'True': true_class,
                    'Pred': pred_class,
                    'Prop': prop
                })

    error_df = pd.DataFrame(error_data).sort_values(by='Prop', ascending=False).head(top_n)

    print("\n🔍 Prop Error (Top {}):".format(top_n))
    print(error_df)

    folder = os.path.dirname(cm_path)
    error_df.to_csv(os.path.join(folder, "prop_error_analysis.csv"), index=False)

    return error_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_mat', type=str, default=None)
    parser.add_argument('--top_n', '-n', type=int, default=30)
    args = parser.parse_args()

    analysis_cm(args.conf_mat, args.top_n)