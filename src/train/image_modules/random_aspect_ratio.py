import random

import torch
from torchvision.transforms import functional


class RandomAspectRatio(torch.nn.Module):
    def __init__(self, ratio_range=(3. / 4., 4. / 3.),
                 ratio_percent=0.5,
                 interpolation=functional.InterpolationMode.BILINEAR):
        torch.nn.Module.__init__(self,)
        self.ratio_range = ratio_range
        self.ratio_percent = int(ratio_percent * 100)
        self.interpolation = interpolation

    def forward(self, img):
        i = torch.randint(0, 100, size=(1,)).item()
        if i >= self.ratio_percent:
            return img

        # 获取当前宽高
        _, h, w = functional.get_dimensions(img)

        # 随机一个宽高比
        aspect_ratio = random.uniform(*self.ratio_range)

        # 计算新的宽高
        if w / h < aspect_ratio:
            new_w = int(h * aspect_ratio)
            new_h = h
        else:
            new_w = w
            new_h = int(w / aspect_ratio)

        # 缩放
        img = functional.resize(img, size=[new_h, new_w], interpolation=self.interpolation)

        return img
