import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import narrow_like_


class AddNoise(nn.Module):
    """Inject single-channel noise whose amplitude and offset depend on the style.

    Adds the noise if ``cat=False``, otherwise concatenates it as an extra channel.
    """

    def __init__(self, cat, chan=1, style_size=1):
        super().__init__()

        self.cat = cat
        self.chan = chan

        self.nn = nn.Sequential(
            nn.Linear(style_size, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Linear(16, 2*chan),
        )

    def forward(self, x, s, noise=None):
        """If ``noise`` is None, a new Gaussian realisation is drawn."""
        b, c, h, w, d = x.shape

        if noise is None:
            noise = torch.randn(b, self.chan, h, w, d, device=x.device)
        else:
            noise = noise.to(x.device)

        scale, bias = self.nn(s).chunk(2, dim=1)
        scale = scale.view(b, -1, 1, 1, 1)
        bias = bias.view(b, -1, 1, 1, 1)
        noise = noise * scale + bias

        if self.cat:
            return torch.cat([x, noise], dim=1)
        return x + noise


class LeakyReLUStyled(nn.Module):
    """Leaky ReLU whose negative slope is predicted from the style vector."""

    def __init__(self, negative_slope=0.2, inplace=False, style_size=5):
        super().__init__()
        self.inplace = inplace
        self.negative_slope = negative_slope

        self.nn = nn.Sequential(
            nn.Linear(style_size, 2*style_size),
            nn.LeakyReLU(0.2),
            nn.Linear(2*style_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, style=None):
        if style is None:
            return F.leaky_relu(x, self.negative_slope, self.inplace)

        slope = self.nn(style).view(-1, 1, 1, 1, 1)
        return torch.where(x >= 0, x, x * slope)


class ConvStyled3d(nn.Module):
    """3D convolution with style modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from ``torch.nn._ConvNd.reset_parameters()``.

    ``resample`` can be None (plain convolution), ``'U'`` (transposed
    convolution, 2x up) or ``'D'`` (strided convolution, 2x down).
    """

    def __init__(self, in_chan, out_chan, style_size, kernel_size=3, stride=1,
                 bias=True, resample=None):
        super().__init__()

        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == 'U':
            K3 = (kernel_size,) * 3
            # transposed-convolution weights are stored as [in, out, ...]
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = stride
            self.conv = F.conv_transpose3d
        elif resample == 'D':
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = 2
            self.conv = F.conv3d
        else:
            raise ValueError('resample type {} not supported'.format(resample))
        self.resample = resample

        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5),
            mode='fan_in',  # effectively 'fan_out' for 'D'
            nonlinearity='leaky_relu',
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        def init_weight(m):
            if type(m) is nn.Linear:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in',
                                         nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.ones_(m.bias)

        self.style_block = nn.Sequential(
            nn.Linear(in_features=style_size, out_features=in_chan),
        )
        self.style_block.apply(init_weight)

    def forward(self, inputs):
        """``inputs`` is a tuple ``(x, s)`` with x [N, Cin, D, H, W] and s [N, style_size]."""
        x, s = inputs[0], inputs[1]
        eps = 1e-8

        N, Cin, *DHWin = x.shape
        C0, C1, *K3 = self.weight.shape

        if self.resample == 'U':
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        # modulation
        s = self.style_block(s)
        if self.resample == 'U':
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s

        # demodulation
        if self.resample == 'U':
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        # grouped convolution applies a different (modulated) kernel per sample
        w = w.reshape(N * C0, C1, *K3)
        x = x.reshape(1, N * Cin, *DHWin)
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)
        _, _, *DHWout = x.shape

        return x.view(N, Cout, *DHWout)


class ResNetBlock(nn.Module):
    """Residual block: [pad, noise, conv, act] x 2, plus a 1x1 styled skip connection."""

    def __init__(self, in_chan, mid_chan, out_chan, style_size, last_act=True, cat_noise=False):
        super().__init__()
        self.last_act = last_act

        self.skip = ConvStyled3d(in_chan, out_chan, style_size, kernel_size=1)

        self.addnoise1 = AddNoise(cat_noise, style_size=style_size)
        self.conv1 = ConvStyled3d(in_chan + cat_noise, mid_chan, style_size, kernel_size=3)
        self.act1 = LeakyReLUStyled(0.2, True)
        self.addnoise2 = AddNoise(cat_noise, style_size=style_size)
        self.conv2 = ConvStyled3d(mid_chan + cat_noise, out_chan, style_size, kernel_size=3)
        self.act2 = LeakyReLUStyled(0.2, True)

        if last_act:
            self.act3 = LeakyReLUStyled(0.2, True)

    def forward(self, x, s, noise1=None, noise2=None):
        y = self.skip((x, s))

        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='reflect')
        x = self.addnoise1(x, s, noise=noise1)
        x = self.conv1((x, s))
        x = self.act1(x, s)

        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='reflect')
        x = self.addnoise2(x, s, noise=noise2)
        x = self.conv2((x, s))
        x = self.act2(x, s)

        x = x + narrow_like_(y, x)

        if self.last_act:
            x = self.act3(x, s)

        return x


class UpBlock(nn.Module):
    """2x trilinear upsampling followed by two styled convolutions."""

    def __init__(self, in_chan, style_size, cat_noise=False):
        super().__init__()
        self.addnoise1 = AddNoise(cat_noise, style_size=style_size)
        self.conv1 = ConvStyled3d(in_chan, in_chan, style_size, kernel_size=3)
        self.act1 = LeakyReLUStyled(0.2, True)
        self.addnoise2 = AddNoise(cat_noise, style_size=style_size)
        self.conv2 = ConvStyled3d(in_chan + cat_noise, in_chan, style_size, kernel_size=3)
        self.act2 = LeakyReLUStyled(0.2, True)

    def forward(self, x, s, noise1=None, noise2=None):
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='reflect')
        x = self.addnoise1(x, s, noise=noise1)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        x = self.conv1((x, s))
        x = self.act1(x, s)
        x = self.addnoise2(x, s, noise=noise2)
        x = self.conv2((x, s))
        x = self.act2(x, s)
        return x


class DownBlock(nn.Module):
    """2x downsampling with a styled stride-2 convolution."""

    def __init__(self, in_chan, style_size, cat_noise=False):
        super().__init__()
        self.addnoise = AddNoise(cat_noise, style_size=style_size)
        self.conv = ConvStyled3d(in_chan + cat_noise, in_chan, style_size, kernel_size=2, stride=2)
        self.act = LeakyReLUStyled(0.2, True)

    def forward(self, x, s, noise=None):
        x = self.addnoise(x, s, noise=noise)
        x = self.conv((x, s))
        x = self.act(x, s)
        return x
