import torch
from torch import nn
from torchvision.models import AlexNet_Weights


class ClassifyToLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
        self.classifier = nn.Sequential(*[alex_net.classifier[i] for i in range(1)])
        self.dense = nn.Sequential(
            nn.Linear(input_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=.1),
        )
        self.label = nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.lstm = nn.LSTM(output_size, output_size, num_layers=4, bidirectional=True, dropout=0.1)
        self.label2 = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, input_tensors):
        x = self.label(self.dense(input_tensors))
        x, _ = self.lstm(x)
        return self.label2(x)

    @staticmethod
    def get_name():
        return "alex_lstm"
