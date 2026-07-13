import torch
import torch.nn as nn
import torch.fft


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y, 2)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x: (batch, channels, x, y)
        x_ft = torch.fft.rfft2(x, norm='ortho')

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        modes1 = min(self.modes1, x_ft.size(-2))
        modes2 = min(self.modes2, x_ft.size(-1))

        # weight layout: (in_channel, out_channel, modes1, modes2, 2)
        w = self.weight[:, :, :modes1, :modes2]
        out_ft[..., :modes1, :modes2] = self.compl_mul2d(x_ft[..., :modes1, :modes2], w)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, modes1=16, modes2=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, x, y, in_channels)
        b, nx, ny, c = x.shape
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, width, nx, ny)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.activation(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.activation(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.activation(x1 + x2)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(in_ch, out_ch, device='cpu'):
    model = FNO2d(in_ch, out_ch)
    return model.to(device)
