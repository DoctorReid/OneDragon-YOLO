from PIL import Image

from train.image_modules.random_aspect_ratio import RandomAspectRatio
from train.image_modules.square_pad import SquarePad


def classify_augmentations(
    size=224,
    ratio=None,
    ratio_percent=0,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation=Image.BILINEAR,
):
    """
    Classification transforms with augmentation for training. Inspired by timm/data/transforms_factory.py.

    Args:
        size (int): image size
        scale (tuple): scale range of the image. default is (0.08, 1.0)
        ratio (tuple): aspect ratio range of the image. default is (3./4., 4./3.)
        mean (tuple): mean values of RGB channels
        std (tuple): std values of RGB channels
        hflip (float): probability of horizontal flip
        vflip (float): probability of vertical flip
        auto_augment (str): auto augmentation policy. can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): image HSV-Hue augmentation (fraction)
        hsv_s (float): image HSV-Saturation augmentation (fraction)
        hsv_v (float): image HSV-Value augmentation (fraction)
        force_color_jitter (bool): force to apply color jitter even if auto augment is enabled
        erasing (float): probability of random erasing
        interpolation (T.InterpolationMode): interpolation mode. default is T.InterpolationMode.BILINEAR.

    Returns:
        (T.Compose): torchvision transforms
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_augmentations() size {size} must be integer, not (list, tuple)")
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    primary_tfl = [
        RandomAspectRatio(ratio, ratio_percent=ratio_percent, interpolation=interpolation),
        SquarePad(after_size=size, interpolation=interpolation),
    ]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


def classify_transforms(
    size=224,
    interpolation=Image.BILINEAR
):
    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")
    import torchvision.transforms as T  # scope for faster 'import ultralytics'
    tfl = [
        SquarePad(),
        T.Resize(size, interpolation=interpolation),
        T.ToTensor(),
    ]
    return T.Compose(tfl)
