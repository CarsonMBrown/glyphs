from torch import nn


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden_size = 100
        self.dense = nn.Sequential(
            nn.Linear(input_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.label = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(self.hidden_size, output_size)
        )

    def forward(self, input_tensors):
        return self.label(self.dense(input_tensors))

    @staticmethod
    def get_name():
        return "linear"
