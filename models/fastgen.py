import torch
from torch import nn
from torch.nn.utils import spectral_norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block

def build_nfc(nf):
    nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v*nf)

    return nfc

class FastGen(nn.Module):
    def __init__(self, ngf=4, nz=64, img_shape=(3,32,32), n_classes=None):
        super(FastGen, self).__init__()
        nc, img_size, _ = img_shape

        if n_classes is None:
            self.label_emb = None
            in_dim = nz
        else:
            self.label_emb = nn.Embedding(n_classes, n_classes)
            in_dim = nz + n_classes

        nfc = build_nfc(ngf)
        self.init = InitLayer(in_dim, channel=nfc[4])
                                
        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])

        self.se_16  = SEBlock(nfc[4], nfc[16])
        self.se_32 = SEBlock(nfc[8], nfc[32])
        
        if img_size == 32:
            self.to_final = conv2d(nfc[32], nc, 3, 1, 1, bias=False) # (i-k+2p)s + 1
        elif img_size == 28:
            self.to_final = conv2d(nfc[32], nc, 5, 1, 0, bias=False)
        
        self.apply(weights_init)
        
    def forward(self, z, labels=None):
        if labels is None:
            gen_input = z
        else:
            gen_input = torch.cat((self.label_emb(labels), z), -1)
        
        feat_4   = self.init(gen_input)
        # print(f'feat_4={feat_4.shape}')
        feat_8   = self.feat_8(feat_4)
        # print(f'feat_8={feat_8.shape}')
        # feat_16  = self.se_16(feat_4, self.feat_16(feat_8))
        t = self.feat_16(feat_8)
        # print(f't_16={t.shape}')
        feat_16  = self.se_16(feat_4, t)
        # print(f'feat_16={feat_16.shape}')
        t = self.feat_32(feat_16)
        # print(f't_32={t.shape}')
        feat_32  = self.se_32(feat_8, t)
        # print(f'feat_32={feat_32.shape}')
        img = self.to_final(feat_32)
        return torch.tanh(img)


if __name__ == '__main__':
    n_classes=10
    net = FastGen(ngf=8, nz=256, img_shape=(1,28,28))
    noise = torch.randn(5,256)
    import numpy as np
    labels = torch.LongTensor(np.random.randint(0, 10, 5))
    # img = net(noise, labels)
    img = net(noise)
    print(img.shape)

    def get_n_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"Generator has n_params={get_n_model_params(net)}M.")
