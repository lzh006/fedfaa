import torch
from torch.autograd import Variable
from torch import nn
import math

class ComOps(nn.Module):
    def __init__(self, nc=3, in_dim=32, n_classes=10, feats_loc=-1):
        super().__init__()
        self.dim_feat = in_dim
        self.n_classes = n_classes
        self.feats_loc = feats_loc
    
    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    def point_grad_to(self, target):
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()
            p.grad.data.add_(p.data - target_p.data)
    
    def point_grad_by_avg(self, cmodels_dict):
        for p in self.parameters():
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()

        for cidx in cmodels_dict:
            target = cmodels_dict[cidx]
            for p, target_p in zip(self.parameters(), target.parameters()):
                p.grad.data.add_((p.data - target_p.data)/len(cmodels_dict))

    