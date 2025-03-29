import argparse
import os
import torch


def convert_compiled_model(model_path):
    compiled_pt = torch.load(model_path, map_location='cpu')

    # 若是完整 checkpoint
    if 'model_state_dict' in compiled_pt :
        state_dict = compiled_pt ['model_state_dict']
    else:
        state_dict = compiled_pt 

    new_state_dict = {
        k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
    }
    new_model_path = os.path.join(os.path.dirname(model_path), "best_model.pth")
    torch.save(new_state_dict, new_model_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./trained_weights/ResNet152/compiled_model.pth')
    args = parser.parse_args()

    convert_compiled_model(args.model_path)