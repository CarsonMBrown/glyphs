from torch import nn


class LinearToLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden_size = 100
        self.dense = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=.1),
        )
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True, dropout=0.1)
        self.label = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, output_size)
        )

    def forward(self, input_tensors):
        x, _ = self.lstm(self.dense(input_tensors))
        return self.label(x)

    @staticmethod
    def get_name():
        return "dense_to_lstm"
