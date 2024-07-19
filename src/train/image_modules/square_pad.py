import torch
from torchvision.transforms import functional


class SquarePad(torch.nn.Module):
    def __init__(self, fill_color=114,
                 after_size: int = None,
                 interpolation=functional.InterpolationMode.BILINEAR):
        """
        按图片的最长边，往图片的下方或右方填充至正方形
        如果传入 after_size，则最终正方形会缩放到这个边长
        """
        torch.nn.Module.__init__(self)
        self.fill_color = fill_color
        self.after_size: int = after_size  # 需要缩放到的边长
        self.interpolation = interpolation

    def forward(self, img):
        # 获取当前宽高
        _, h, w = functional.get_dimensions(img)
        if self.after_size is None:
            if h == w:
                return img
            else:
                max_dim = max(h, w)
                return functional.pad(img, padding=[0, 0, max_dim - w, max_dim - h], fill=self.fill_color)
        else:
            if h > w:
                new_h = self.after_size
                new_w = int(new_h / h * w)
            elif w > h:
                new_w = self.after_size
                new_h = int(new_w / w * h)
            else:
                new_h = self.after_size
                new_w = self.after_size

            resize = functional.resize(img, [new_h, new_w], interpolation=self.interpolation)
            return functional.pad(resize,
                                  padding=[0, 0, self.after_size - new_w, self.after_size - new_h],
                                  fill=self.fill_color)
