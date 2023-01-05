from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden_size = 50
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=1, bidirectional=True)
        self.label = nn.Sequential(
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size * 2, output_size)
        )

    def forward(self, input_tensors):
        x, _ = self.lstm(input_tensors)
        return self.label(x)

    @staticmethod
    def get_name():
        return "simple_lstm"
