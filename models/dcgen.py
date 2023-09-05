from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from copy import deepcopy
import os

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class DCGen(nn.Module):
    def __init__(self, ngf=64, nz=100, img_shape=(3,32,32), n_classes=None):
        super(DCGen, self).__init__()
        nc, img_size, _ = img_shape

        self.init_size = img_size // 4

        if n_classes is None:
            self.label_emb = None
            in_dim = nz
        else:
            self.label_emb = nn.Embedding(n_classes, n_classes)
            in_dim = nz + n_classes

        self.l1 = nn.Linear(in_dim, ngf * 2 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),  
        )

        self.apply(weights_init_normal)

    def forward(self, z, labels=None):
        if labels is None:
            gen_input = z
        else:
            gen_input = torch.cat((self.label_emb(labels), z), -1)

        out = self.l1(gen_input)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        return self.conv_blocks(out)


import torch
if __name__ == '__main__':
    def get_n_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # gan = DCGen(nz=484, img_shape=(1,28,28), n_classes=10)
    # print(f"cgan has n_params={get_n_model_params(gan)}M.")

    # z = torch.randn(5,484)
    # label = torch.LongTensor([1,2,3,4,5])
    # imgs = gan(z, label)
    # print(imgs.shape)
