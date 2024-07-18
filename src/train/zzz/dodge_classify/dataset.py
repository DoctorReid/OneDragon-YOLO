from ultralytics.data import ClassificationDataset

from train.zzz.dodge_classify.transforms import classify_augmentations, classify_transforms


class DodgeDataset(ClassificationDataset):

    def __init__(self, root, args, augment=False, prefix=""):
        ClassificationDataset.__init__(self, root=root, args=args, augment=augment, prefix=prefix)

        # 覆盖原有的
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                ratio=(5./4., 16./9.),
                ratio_percent=args.scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )
