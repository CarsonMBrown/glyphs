from torch import nn


class LSTMtoLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden_size = 100
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=4, bidirectional=True, dropout=0.1)
        self.label = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size)
        )

    def forward(self, input_tensors):
        x, _ = self.lstm(input_tensors)
        return self.label(x)

    @staticmethod
    def get_name():
        return "lstm_to_linear"
