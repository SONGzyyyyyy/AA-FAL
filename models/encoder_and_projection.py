from torch import nn
import torch.nn.functional as F
from models.mlp_head import MLPHead
from models.complexcnn import ComplexConv

class Encoder_and_projection(nn.Module):
    def __init__(self, projection_head, num_layers=9, in_channels=1, out_channels=64, kernel_size=4, stride=2, fc_dim=1024):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ComplexConv(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride))
            layers.append(nn.BatchNorm1d(128))  # ComplexConv 输出实部+虚部拼接为 2*out_channels
            layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(fc_dim)
        self.projection = MLPHead(in_channels=fc_dim, **projection_head)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        embedding = F.relu(x)
        projection = self.projection(embedding)
        return embedding, projection