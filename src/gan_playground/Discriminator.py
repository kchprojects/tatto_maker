import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, input_size: int, channels=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(
                channels, input_size, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        self.layers.append(nn.BatchNorm2d(input_size))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        real_size = input_size // 2
        while real_size > 4:
            output_size = input_size * 2
            self.layers.append(
                nn.Conv2d(
                    input_size,
                    output_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.layers.append(nn.BatchNorm2d(output_size))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_size = output_size
            real_size //= 2
        self.layers.append(
            nn.Conv2d(input_size, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        data = x
        for l in self.layers:
            data = l(data)
        return self.activation(self.flatten(data))
