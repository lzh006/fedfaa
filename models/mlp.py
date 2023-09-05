from torch import nn

class MLP(nn.Module):
    def __init__(self, n_classes=2):
        super(MLP, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, feats_mode='off'):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        feats = x
        logits = self.fc3(x)
        return (logits, feats) if feats_mode=='also' else logits