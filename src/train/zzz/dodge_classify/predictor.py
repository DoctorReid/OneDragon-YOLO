from ultralytics.models.yolo.classify import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG

from train.zzz.dodge_classify.transforms import classify_transforms


class DodgePredictor(ClassificationPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        ClassificationPredictor.__init__(
            self,
            cfg=cfg,
            overrides=overrides,
            _callbacks=_callbacks
        )

    def setup_source(self, source):
        ClassificationPredictor.setup_source(self, source=source)

        self.transforms = classify_transforms(self.imgsz[0])
