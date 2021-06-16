import torch
import torch.nn as nn
from torchsummary import summary

VGG16 = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]  # Then we flatten and 4096x4096x1000 Linear Layers


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes, architechture):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layer(architechture)
        self.fcs = nn.Sequential(
            nn.Linear(
                512 * 7 * 7, 4096
            ),  # eval(224 / 2**5) ==7 |(5 comes from 5 max pool layers)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layer(self, architechture):
        layers = []
        in_channels = self.in_channels

        for x in architechture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),  # not in the paper but why not!
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG(in_channels=3, num_classes=1000, architechture=VGG16).to(device)
    x = torch.randn(1, 3, 244, 244)
    print(model(x).shape)
    summary(model, input_size=(3, 244, 244))
