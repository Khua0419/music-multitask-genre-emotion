import torch, torch.nn as nn

class Backbone2D(nn.Module):
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),   nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),  nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)))
        self.proj = nn.Linear(128, feat_dim)
    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)

class MultiTaskNet(nn.Module):
    def __init__(self, in_ch=3, feat_dim=128, num_genres=3, emo_dim=2):
        super().__init__()
        self.backbone = Backbone2D(in_ch, feat_dim)
        self.g_head = nn.Linear(feat_dim, num_genres)
        self.e_head = nn.Linear(feat_dim, emo_dim)
    def forward(self, x):
        z = self.backbone(x)
        return self.g_head(z), self.e_head(z)
