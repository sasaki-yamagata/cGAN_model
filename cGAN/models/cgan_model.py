import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        # self.fc1 = nn.Linear(n_input, 600)
        # self.norm = nn.BatchNorm1d(600)
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.fc2 = nn.Linear(600, 1)
        self.dis = nn.Sequential(
            nn.Linear(n_input, 600),
            nn.BatchNorm1d(600),
            nn.LeakyReLU(inplace=True),
            nn.Linear(600, 1)
        )
    
    def forward(self, x):
        x = self.dis(x)

        # print(x.shape)
        # x = self.fc1(x)
        # x = self.norm(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(n_input, 700),
            nn.BatchNorm1d(700),
            nn.LeakyReLU(inplace=True),
            nn.Linear(700, n_output)
        )
    
    def forward(self, x):
        return self.gen(x)


