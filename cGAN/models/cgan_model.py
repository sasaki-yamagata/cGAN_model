import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_input):
        self.dis = nn.Sequential(
            nn.Linear(n_input, 600),
            nn.BatchNorm2d(600),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(600, 1)
        )
    
    def forward(self, x):
        return self.dis(x)


class Generator(nn.Module):
    def __init__(self, n_input, n_output):
        self.gen = nn.Sequential(
            nn.Linear(n_input, 700),
            nn.BatchNorm2d(700),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(700, n_output)
        )
    
    def forward(self, x):
        return self.gen(x)

    