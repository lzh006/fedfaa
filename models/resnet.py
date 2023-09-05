import math, torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from .comops import ComOps

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def norm2d(gnnp, planes):
    if gnnp is not None and gnnp > 0:
        # gnnp == planes -> InstanceNorm
        # gnnp == 1 -> LayerNorm
        return nn.GroupNorm(gnnp, planes)
    else:
        return nn.BatchNorm2d(planes)

class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1
    
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, gnnp=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.layers = nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            norm2d(gnnp, planes=out_planes),
            nn.ReLU(inplace=True),

            conv3x3(out_planes, out_planes),
            norm2d(gnnp, planes=out_planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.layers(x) + residual
        return self.relu(out)


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, gnnp=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            norm2d(gnnp, planes=out_planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(gnnp, planes=out_planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_planes, out_channels=out_planes * Bottleneck.expansion, kernel_size=1, bias=False),
            norm2d(gnnp, planes=out_planes * Bottleneck.expansion)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.layers(x) + residual
        return self.relu(out)

class ResNet(ComOps):
    def __init__(self, nc=3, in_dim=32, n_classes=10, feats_loc=-1, size=8, gnnp=None, scaling=1):
        super().__init__(nc, in_dim, n_classes, feats_loc)

        if size % 6 != 2:
            raise ValueError("size must be 6n + 2:", size)
        n_blocks = (size - 2) // 6
        block_fn = Bottleneck if size >= 44 else BasicBlock

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.init_block = nn.Sequential(
            nn.Conv2d(nc, 16 * scaling, kernel_size=3, stride=1, padding=1, bias=False),
            norm2d(gnnp, planes=int(16 * scaling)),
            nn.ReLU(inplace=True)
        )
        self.block1 = self._make_block(block_fn, int(16 * scaling), n_blocks, gnnp=gnnp)
        self.block2 = self._make_block(block_fn, int(32 * scaling), n_blocks, stride=2, gnnp=gnnp)
        self.block3 = nn.Sequential(
            self._make_block(block_fn, int(64 * scaling), n_blocks, stride=2, gnnp=gnnp),
            nn.AvgPool2d(kernel_size=8)
        )

        self.dim_feat = (self.dim_feat // 8)**2 * self.inplanes
        self.classifier = nn.Linear(self.dim_feat, n_classes)

        self._weight_initialization()

    def _make_block(self, block_fn, planes, n_blocks, stride=1, gnnp=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_fn.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(gnnp, planes=planes * block_fn.expansion),
            )

        layers = [block_fn(self.inplanes, planes, stride, downsample, gnnp)]
        self.inplanes = planes * block_fn.expansion
        self.dim_feat = (self.dim_feat -1) // stride + 1
        layers.extend(block_fn(self.inplanes, planes, gnnp=gnnp) for _ in range(1, n_blocks))

        return nn.Sequential(*layers)
    

    def forward(self, x, feats_mode='off'):
        x = self.init_block(x)
        if self.feats_loc == -4: feats = x.reshape(x.size(0), -1)
        if feats_mode == 'only': return None, feats

        x = self.block1(x)
        if self.feats_loc == -3: feats = x.reshape(x.size(0), -1)
        if feats_mode == 'only': return None, feats

        x = self.block2(x)
        if self.feats_loc == -2: feats = x.reshape(x.size(0), -1)
        if feats_mode == 'only': return None, feats

        x = self.block3(x)
        x = x.view(x.size(0), -1)
        if self.feats_loc == -1: feats = x
        if feats_mode == 'only': return None, feats
            
        logits = self.classifier(x)
        return (logits, feats) if feats_mode=='also' else logits
    


if __name__ == '__main__':
    net = ResNet(nc=3, in_dim=32, size=20, n_classes=10)
    imgs = torch.randn(5,3,32,32)
    logits, feats = net(imgs, feats_also=True)
    print(f'logits: {logits.shape}, feats: {feats.shape}')