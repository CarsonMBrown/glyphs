
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import AlexNet_Weights


class AlexNet(nn.Module):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomApply(nn.ModuleList([
        #     transforms.RandomAdjustSharpness()
        # ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, input_size, output_size):
        super().__init__()
        alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = alex_net.features
        self.avgpool = alex_net.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=.1),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
        )
        self.lstm = nn.LSTM(output_size, output_size, num_layers=2, bidirectional=True, dropout=0.1)
        self.label = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.label(x)

    @staticmethod
    def get_name():
        return "alex"
