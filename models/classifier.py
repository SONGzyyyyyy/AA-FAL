from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 6)

    def forward(self, x):
        return self.linear(x)