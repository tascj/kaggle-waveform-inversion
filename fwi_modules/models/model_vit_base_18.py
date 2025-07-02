import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


def process_input(x):
    # x: (N, 5, 1000, 70)
    N = x.size(0)
    x = x.reshape(N, 5, 250, 4, 70)
    x = x.permute(0, 1, 2, 4, 3)
    x = x.reshape(N, 5, 250, 280)
    x = F.interpolate(x, size=(288, 288), mode="bilinear", align_corners=False)
    return x


class FWIModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = timm.create_model(
            "eva02_base_patch16_clip_224",
            pretrained=pretrained,
            dynamic_img_size=True,
            in_chans=5,
        )
        self.backbone = backbone
        self.head = nn.Linear(768, 16)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x):
        N = x.size(0)
        x = process_input(x)
        x = self.backbone.forward_features(x)
        x = x[:, 1:]
        x = self.head(x)
        x = x.permute(0, 2, 1).reshape(N, 16, 18, 18)
        x = self.pixel_shuffle(x)
        x = x[:, :, 1:-1, 1:-1]

        x = F.sigmoid(x.float()) * 6000
        return x


if __name__ == "__main__":
    x = torch.randn(2, 5, 1000, 70)
    model = FWIModel()
    with torch.no_grad():
        y = model(x)
    print(y.shape)
