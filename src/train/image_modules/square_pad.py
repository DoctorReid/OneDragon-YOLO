import torch
from torchvision.transforms import functional


class SquarePad(torch.nn.Module):
    def __init__(self, fill_color=114):
        torch.nn.Module.__init__(self)
        self.fill_color = fill_color

    def forward(self, img):
        # 获取当前宽高
        _, h, w = functional.get_dimensions(img)
        if h == w:
            return img
        else:
            max_dim = max(h, w)
            return functional.pad(img, padding=[0, 0, max_dim - w, max_dim - h], fill=self.fill_color)
