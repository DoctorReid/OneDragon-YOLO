from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import DEFAULT_CFG

from train.zzz.dodge_classify.dataset import DodgeDataset


class DodgeTrainer(ClassificationTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        ClassificationTrainer.__init__(self, cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        return DodgeDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)
