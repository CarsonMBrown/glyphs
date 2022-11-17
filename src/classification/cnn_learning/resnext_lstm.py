from torch import nn, randn
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, resnext101_32x8d


class AddGaussianNoise(object):
    """
    Author: Ptrblck
    https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ResNext50LSTM(nn.Module):
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(.1, .1), shear=10),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_classify = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomAdjustSharpness(2, 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_size, output_size):
        super().__init__()
        self.resnext = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.resnext.fc = nn.Sequential(
            nn.LSTM(self.resnext.fc.in_features, output_size, num_layers=1, bidirectional=True, dropout=0.1),
        )
        self.classifier = nn.Linear(output_size * 2, output_size)

    def forward(self, x):
        x, _ = self.resnext(x)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "resnext50_lstm"


# EPOCH 19: PRECISION 0.8248833333333341 RECALL 0.8184 FSCORE 0.8106371428571422
# EPOCH 28: PRECISION 0.847766666666667 RECALL 0.8324 FSCORE 0.82944
# EPOCH 41: PRECISION 0.852183333333334 RECALL 0.8412 FSCORE 0.8367676190476188
# EPOCH 56: PRECISION 0.8464500000000006 RECALL 0.839 FSCORE 0.833355238095238
# Start training at 74 with higher dropout (.1 -> .33)
class ResNextLongLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        dropout = .33
        self.resnext = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.resnext.fc = nn.LSTM(self.resnext.fc.in_features, output_size * 2, num_layers=1, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(output_size * 4, output_size * 4),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(output_size * 4, output_size),
        )

    def forward(self, x):
        x, _ = self.resnext(x)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "resnext50_long_lstm"


class ResNextDeepLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        dropout = .5
        self.resnext = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.resnext.fc = nn.Sequential(
            nn.LSTM(self.resnext.fc.in_features, output_size * 4, num_layers=1, bidirectional=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(output_size * 8, output_size * 4),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(output_size * 4, output_size),
        )

    def forward(self, x):
        x, _ = self.resnext(x)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "resnext50_deep_lstm"


# epoch 73: FSCORE 0.7000923809523812
class ResNextClassifyLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.resnext = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.resnext.fc = nn.Sequential(
            nn.Linear(self.resnext.fc.in_features, 200),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LSTM(100, output_size, num_layers=1, bidirectional=True, dropout=0.1)
        )
        self.classifier = nn.Linear(output_size * 2, output_size)

    def forward(self, x):
        x, _ = self.resnext(x)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "resnext_classify_lstm"


# EPOCH 36:
#   batch 250 loss: 0.5359474135190249
#   batch 500 loss: 0.484837262943387
# LOSS train 0.484837262943387 valid 0.9876185059547424
# PRECISION 0.720560000000001 RECALL 0.722 FSCORE 0.7055033333333333
# LR = .001
class ResNext101LSTM(nn.Module):
    transform_train = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomPerspective(distortion_scale=0.2, p=.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, shear=10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
        transforms.RandomAdjustSharpness(1.2, p=.1),
        transforms.RandomAdjustSharpness(1.1, p=.1),
        transforms.RandomAdjustSharpness(.9, p=.1),
        transforms.RandomAdjustSharpness(.8, p=.1),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_classify = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_size, output_size):
        super().__init__()
        self.resnext = resnext101_32x8d(ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        self.resnext.fc = nn.LSTM(self.resnext.fc.in_features, output_size * 1, num_layers=1, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=.2),
            nn.ReLU(),
            nn.Linear(output_size * 2, output_size * 2),
            nn.Dropout(p=.2),
            nn.ReLU(),
            nn.Linear(output_size * 2, output_size)
        )

        self.h_0, self.c_0 = None, None

    def forward(self, x):
        x, _ = self.resnext(x)
        return self.classifier(x)

    @staticmethod
    def get_name():
        return "resnext101_lstm"
