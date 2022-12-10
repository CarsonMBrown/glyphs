import torch.nn.functional as F
from torch import nn
from torchvision import transforms


# 0.6615824081012569 32,color
class MNISTCNN(nn.Module):
    """
    BASED ON:
    https://github.com/cdeotte/MNIST-CNN-99.75/blob/master/CNN.ipynb
    https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99
    """
    input_size = 32
    transform_train = transforms.Compose([
        transforms.Resize(input_size - 1, max_size=input_size),
        transforms.Pad(input_size - 1),
        transforms.RandomPerspective(.1),
        transforms.RandomAffine(degrees=20, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=.1),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_train_2 = transforms.Compose([
        transforms.Resize(input_size - 1, max_size=input_size),
        transforms.Pad(input_size - 1, fill=(255, 255, 255)),
        transforms.RandomPerspective(.1),
        transforms.RandomAffine(degrees=20, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=.1),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_train_3 = transforms.Compose([
        transforms.Resize(input_size - 1, max_size=input_size),
        transforms.Pad(input_size - 1, fill=(255, 255, 255)),
        transforms.RandomPerspective(.2),
        transforms.RandomAffine(degrees=15, translate=(0.4, 0.4), scale=(0.6, 1.4), shear=0.4),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=.3),
        transforms.RandomAutocontrast(p=.5),
        transforms.RandomAdjustSharpness(sharpness_factor=.8, p=.1),
        transforms.RandomAdjustSharpness(sharpness_factor=.9, p=.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=.1),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_classify = transforms.Compose([
        transforms.Resize(input_size - 1, max_size=input_size),
        transforms.Pad(input_size - 1),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_classify_2 = transforms.Compose([
        transforms.Resize(input_size - 1, max_size=input_size),
        transforms.Pad(input_size - 1, fill=(255, 255, 255)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dropout = .5
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def get_name():
        return "mnist"


class MNISTCNN_LSTM(nn.Module):
    """
    BASED ON:
    https://github.com/cdeotte/MNIST-CNN-99.75/blob/master/CNN.ipynb
    https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99
    """
    input_size = 32

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dropout = .5
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.lstm = nn.LSTM(128, output_size, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(output_size * 2, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def get_name():
        return "mnist_lstm"


class MNISTCNN_DEEP_LSTM(nn.Module):
    """
    BASED ON:
    https://github.com/cdeotte/MNIST-CNN-99.75/blob/master/CNN.ipynb
    https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99
    """
    input_size = 32

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dropout = .5
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.lstm = nn.LSTM(128, output_size, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(output_size * 2, output_size * 2),
            nn.Linear(output_size * 2, output_size * 2),
            nn.Linear(output_size * 2, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def get_name():
        return "mnist_deep_lstm"

class MNISTCNN_DEEP_ACTIVATED_LSTM(nn.Module):
    """
    BASED ON:
    https://github.com/cdeotte/MNIST-CNN-99.75/blob/master/CNN.ipynb
    https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99
    """
    input_size = 32

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dropout = 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout)
        )

        self.lstm = nn.LSTM(128, output_size, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(output_size * 2, output_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(output_size * 2, output_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(output_size * 2, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def get_name():
        return "mnist_deep_activated_lstm"