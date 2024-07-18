from ultralytics.models.yolo.classify import ClassificationValidator

from train.zzz.dodge_classify.dataset import DodgeDataset


class DodgeValidator(ClassificationValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        ClassificationValidator.__init__(
            self,
            dataloader=dataloader,
            save_dir=save_dir,
            pbar=pbar,
            args=args,
            _callbacks=_callbacks
        )

    def build_dataset(self, img_path):
        return DodgeDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)
