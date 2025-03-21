from .model import ClassificationModel


class HierarchicalClassificationModel(ClassificationModel):
    def __init__(self, config, text_encoding_path, runs_folder=None):
        super().__init__(config, runs_folder)
        
        self.model = None
        self.cate_model = self.load_model(self.config.NUM_CATE_CLASSES)
        self.cate_criterion = self.load_criterion(self.config.LOSS)


    def _train_cate_model(self):
        self.model.train()
        

# TODO Redesign _train(), _val()

