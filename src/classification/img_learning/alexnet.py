import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features, self.avgpool = alex_net.features, alex_net.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, input_img):
        x = torch.flatten(self.avgpool(self.features(input_img)), 1)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "alex"
