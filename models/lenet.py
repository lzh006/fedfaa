from .comops import ComOps
from torch import nn

class LeNet(ComOps):
    def __init__(self, nc=3, in_dim=32, n_classes=10, feats_loc=-1):
        super().__init__(nc, in_dim, n_classes, feats_loc)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(nc, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.dim_feat = self.dim_feat - 4
        self.dim_feat = (self.dim_feat - 2) // 2 + 1

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.dim_feat = self.dim_feat - 4
        self.dim_feat = (self.dim_feat - 2) // 2 + 1

        self.fcs = nn.Sequential(
            nn.Linear(16 * self.dim_feat**2, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )

        self.dim_feat = 84
        self.classifier = nn.Linear(84, n_classes)

        self._weight_initialization()

    def forward(self, x, feats_mode='off'):
        x = self.conv_block1(x)
        if self.feats_loc == -3: feats = x.reshape(x.size(0), -1)
        if feats_mode == 'only': return None, feats

        x = self.conv_block2(x)
        x = x.reshape(x.size(0), -1)
        if self.feats_loc == -2: feats = x
        if feats_mode == 'only': return None, feats
        
        x = self.fcs(x)
        if self.feats_loc == -1: feats = x
        if feats_mode == 'only': return None, feats

        logits = self.classifier(x)

        return (logits, feats) if feats_mode=='also' else logits