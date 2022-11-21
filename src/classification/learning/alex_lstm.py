import torch
from torch import nn
from torchvision import transforms


class AlexNetLSTM(nn.Module):
    preprocess = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(.1,.1), shear=10),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomAdjustSharpness(.5, .5),
        transforms.RandomAdjustSharpness(2, .5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_size, output_size):
        super().__init__()
        # alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
        # self.features = alex_net.features
        # self.avgpool = alex_net.avgpool
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=.1),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
        )
        self.lstm = nn.LSTM(2048, output_size, num_layers=2, bidirectional=True, dropout=0.1)
        self.label = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x, _ = self.lstm(x)
        return self.label(x)

    @staticmethod
    def get_name():
        return "alex_lstm"
