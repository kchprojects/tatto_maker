import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, out_size, channel_count, depth, latent_size=128):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    latent_size,
                    out_size * (2 ** depth),
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_size * (2 ** depth)),
                nn.ReLU(True),
            ]
        )
        for d in range(depth - 1, -1, -1):
            self.layers.append(
                nn.ConvTranspose2d(
                    out_size * (2 ** (d + 1)),
                    out_size * (2 ** d),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.layers.append(nn.BatchNorm2d(out_size * (2 ** d)))
            self.layers.append(nn.ReLU(True))

        self.layers.append(
            nn.ConvTranspose2d(
                out_size, channel_count, kernel_size=4, stride=2, padding=1, bias=False
            )
        )

        self.activation = nn.Tanh()

    def forward(self, x):
        data = x
        for l in self.layers:
            data = l(data)
        return self.activation(data)
