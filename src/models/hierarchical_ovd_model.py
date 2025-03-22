import importlib
from pandas import DataFrame

import src
from utils.utils import coarse_to_fine
from .hierarchical_model import HierarchicalClassificationModel
from .image_encoder import ImageEncoder
from utils import *


class HierarchicalOVDModel(HierarchicalClassificationModel):
    def __init__(self, config, coarse_text_encoding_path, fine_text_encoding_path, runs_folder=None):
        super().__init__(config, runs_folder)

        if config.HIER_MODEL.COARSE_OVD:
            self.coarse_image_encoder = ImageEncoder(self.model)

        if config.HIER_MODEL.FINE_OVD:
            for coarse_class in self.fine_models.keys():
                self.fine_models[coarse_class] = self.load_model(
                    self.config.HIER_MODEL.FINE_MODEL,
                    512
                )
                self.fine_models[coarse_class] = ImageEncoder(self.fine_models[coarse_class])

