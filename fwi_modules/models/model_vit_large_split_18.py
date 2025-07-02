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
    x = F.interpolate(x, size=(252, 252), mode="bilinear", align_corners=False)
    return x


class FWIModel(nn.Module):
    def __init__(self, pretrained=True, split_at=4):
        super().__init__()
        backbone = timm.create_model(
            "eva02_large_patch14_clip_224",
            pretrained=pretrained,
            dynamic_img_size=True,
            in_chans=1,
        )
        self.backbone = backbone
        self.head = nn.Linear(1024, 16)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.split_at = split_at

    def forward(self, x):
        x = process_input(x)
        N, C, H, W = x.size()
        # x = self.backbone.forward_features(x)
        x = self.backbone.patch_embed(x.reshape(N * C, 1, H, W))
        x, rot_pos_embed = self.backbone._pos_embed(x)
        for blk in self.backbone.blocks[: self.split_at]:
            x = blk(x, rope=rot_pos_embed)
        x = x.reshape(N, 5, *x.size()[1:])
        x = x.mean(1)
        for blk in self.backbone.blocks[self.split_at :]:
            x = blk(x, rope=rot_pos_embed)
        x = self.backbone.norm(x)

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
