from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel

from train.zzz.dodge_classify.trainer import DodgeTrainer
from train.zzz.dodge_classify.validator import DodgeValidator


class DodgeYolo(Model):

    def __init__(self, model="yolov8n-cls.pt", task='classify', verbose=False):
        """
        自定义的一个训练闪避模型
        """
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": DodgeTrainer,
                "validator": DodgeValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            }
        }
