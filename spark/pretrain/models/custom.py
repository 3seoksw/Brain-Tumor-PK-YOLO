# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model
from repvit import common


class YourConvNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mlist = nn.ModuleList(
            [
                common.Silence(),
                common.Bbackbone(),
            ]
        )
        self.d0 = common.Down0(64)
        self.d1 = common.Down1(128)
        self.d2 = common.Down2(256)
        self.d3 = common.Down3(512)
        self.d4 = common.Down4(1024)
        self.alld = [self.d0, self.d1, self.d2, self.d3, self.d4]
        self.cblinear1 = common.CBLinear(64, [64])
        self.cblinear3 = common.CBLinear(128, [64, 128])
        self.cblinear5 = common.CBLinear(256, [64, 128, 256])
        self.cblinear7 = common.CBLinear(512, [64, 128, 256, 512])
        self.cblinear9 = common.CBLinear(1024, [64, 128, 256, 512, 1024])
        self.allcblinear = [
            self.cblinear1,
            self.cblinear3,
            self.cblinear5,
            self.cblinear7,
            self.cblinear9,
        ]
        # # conv down 1
        self.conv1 = common.Conv(3, 64, 3, 2)
        self.cbfuse1 = common.CBFuse([0, 0, 0, 0, 0])

        # conv down 2
        self.conv2 = common.Conv(64, 128, 3, 2)
        self.cbfuse2 = common.CBFuse([1, 1, 1, 1])
        self.rep2 = common.RepNCSPELAN4(128, 256, 128, 64, 2)
        # avg-conv down fuse 1
        self.adown3 = common.ADown(256, 256)
        self.cbfuse3 = common.CBFuse([2, 2, 2])
        self.rep3 = common.RepNCSPELAN4(256, 512, 256, 128, 2)

        # avg-conv down fuse 2
        self.adown4 = common.ADown(512, 512)
        self.cbfuse4 = common.CBFuse([3, 3])
        self.rep4 = common.RepNCSPELAN4(512, 1024, 512, 256, 2)

        # avg-conv down fuse 3
        self.adown5 = common.ADown(1024, 1024)
        self.cbfuse5 = common.CBFuse([4])
        self.rep5 = common.RepNCSPELAN4(1024, 1024, 512, 256, 2)

    def get_downsample_ratio(self) -> int:
        return 32

    def get_feature_map_channels(self) -> List[int]:
        return [256, 512, 1024, 1024]

    def forward(self, x: torch.Tensor, hierarchical=False):
        if hierarchical:
            origin = x.clone()
            ls = []
            tmp = []
            bx = None
            for index, modules in enumerate(self.mlist):
                x = modules(x)
                if index == 1:
                    bx = x
            for i in range(5):
                tmp.append(self.allcblinear[i](self.alld[i](bx)))

            fuse1 = self.cbfuse1(
                [tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], self.conv1(origin)]
            )
            fuse2 = self.cbfuse2([tmp[1], tmp[2], tmp[3], tmp[4], self.conv2(fuse1)])
            fuse2 = self.rep2(fuse2)

            fuse3 = self.cbfuse3([tmp[2], tmp[3], tmp[4], self.adown3(fuse2)])
            fuse3 = self.rep3(fuse3)

            fuse4 = self.cbfuse4([tmp[3], tmp[4], self.adown4(fuse3)])
            fuse4 = self.rep4(fuse4)

            fuse5 = self.cbfuse5([tmp[4], self.adown5(fuse4)])
            fuse5 = self.rep5(fuse5)

            ls.append(fuse2)
            ls.append(fuse3)
            ls.append(fuse4)
            ls.append(fuse5)
            return ls
        else:
            for modules in self.mlist:
                x = modules(x)
        return x


@register_model
def V9back(pretrained=False, **kwargs):
    return YourConvNet(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model

    cnn = create_model("V9back")
    print("get_downsample_ratio:", cnn.get_downsample_ratio())
    print("get_feature_map_channels:", cnn.get_feature_map_channels())

    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == "__main__":
    convnet_test()
