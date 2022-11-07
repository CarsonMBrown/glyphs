from torch import nn
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class ResNetLSTM(nn.Module):
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(.1, .1), shear=10),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomAdjustSharpness(2, 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_classify = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomAdjustSharpness(2, 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_size, output_size):
        super().__init__()
        self.resnext = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        # self.resnext = resnext101_64x4d()
        self.resnext.fc = nn.Sequential(
            nn.LSTM(self.resnext.fc.in_features, output_size, num_layers=1, bidirectional=True, dropout=0.1),
        )
        self.classifier = nn.Linear(output_size * 2, output_size)

    def forward(self, x):
        x, _ = self.resnext(x)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "resnext_lstm_sharp2"
