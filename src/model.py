import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResNetBlock, DownBlock, UpBlock, ConvStyled3d
from .utils import narrow_like_


# Number of noise inputs consumed by UNetG.forward (one per AddNoise layer)
N_NOISE = 18


def noise_sizes(chunk_size):
    """Spatial size of each of the ``N_NOISE`` noise fields for a padded input chunk.

    ``chunk_size`` is the side length of the (padded) chunk fed to the network,
    e.g. 256 / 4 + 2*3 = 70 for a 256^3 box with ``nsplit=4, pad=3``, which
    gives [142, 142, 140, 72, 72, 70, 37, ...].
    """
    L = 2 * chunk_size  # after the initial 2x interpolation
    if L % 4:
        raise ValueError(f"Padded chunk size must be even, got {chunk_size}")
    return ([L + 2, L + 2, L]                       # resblock1, down1
            + [L//2 + 2, L//2 + 2, L//2]            # resblock2, down2
            + [L//4 + 2] * 5                        # bottleneck1, bottleneck2, up1 (1st)
            + [L//2 + 2] * 4                        # up1 (2nd), resblock3, up2 (1st)
            + [L + 2] * 3)                          # up2 (2nd), resblock4


def make_noise(chunk_size, seed=None):
    """Draw one fixed noise realisation for ``UNetG`` (list of ``N_NOISE`` tensors)."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()
    return [torch.randn(1, 1, n, n, n, generator=gen) for n in noise_sizes(chunk_size)]


class UNetG(nn.Module):
    """Style-conditioned 3D U-Net generator that super-resolves displacement fields by 2x.

    The input is first trilinearly upsampled by 2x, and the network predicts a
    residual correction on top of it. Every convolution and activation is
    modulated by the style vector ``s`` (cosmological parameters), and noise is
    injected after each block (StyleGAN2-like).

    Architecture adapted from
    https://github.com/dschaurecker/dl_halo/blob/main/train_test/map2map/models/generator.py

    Parameters
    ----------
    in_chan, out_chan : int
        Number of input / output channels (3 for a displacement field).
    style_size : int
        Length of the style vector.
    chan_base : int
        Number of channels after the input convolution; deeper levels use
        multiples of this. Must be 32 for the released weights.

    Notes
    -----
    The field is processed in periodic chunks: a chunk of size ``N`` padded by
    ``pad`` cells on each side gives an output of ``2*(N + 2*pad)`` per side,
    whose inaccurate edges are cropped to ``2*N`` (see ``utils.chunk_upscale``).
    """

    def __init__(self, in_chan, out_chan, style_size, chan_base=32):
        super().__init__()
        c = chan_base

        self.conv_in = ConvStyled3d(in_chan, c, style_size, kernel_size=1, stride=1)

        self.resblock1 = ResNetBlock(c, 2*c, 2*c, style_size)
        self.down1 = DownBlock(2*c, style_size)

        self.resblock2 = ResNetBlock(2*c, 4*c, 4*c, style_size)
        self.down2 = DownBlock(4*c, style_size)

        self.bottleneck1 = ResNetBlock(4*c, 8*c, 16*c, style_size)
        self.bottleneck2 = ResNetBlock(16*c, 8*c, 4*c, style_size)

        self.up1 = UpBlock(4*c, style_size)
        self.resblock3 = ResNetBlock(4*c + 4*c, 4*c, 2*c, style_size)

        self.up2 = UpBlock(2*c, style_size)
        self.resblock4 = ResNetBlock(2*c + 2*c, 2*c, c, style_size, last_act=False)

        self.conv_out = ConvStyled3d(c, out_chan, style_size, kernel_size=1, stride=1)

    def forward(self, x, s, noise_arr=None):
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, in_chan, N, N, N]
            Normalised low-resolution displacement field (padded chunk).
        s : torch.Tensor, shape [B, style_size]
            Style vector ``[a, Om_k, sigma8, Om_m, h]``.
        noise_arr : list of torch.Tensor or None, length ``N_NOISE``
            Fixed noise for each noise-injection layer. Entries that are
            ``None`` (or ``noise_arr=None``) are drawn randomly on the fly.

        Returns
        -------
        torch.Tensor, shape [B, out_chan, 2N, 2N, 2N]
        """
        if noise_arr is None:
            noise_arr = [None] * N_NOISE

        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = x   # the network predicts the residual to the interpolated field

        x = self.conv_in((x, s))

        x1 = self.resblock1(x, s, noise1=noise_arr[0], noise2=noise_arr[1])
        x = self.down1(x1, s, noise=noise_arr[2])

        x2 = self.resblock2(x, s, noise1=noise_arr[3], noise2=noise_arr[4])
        x = self.down2(x2, s, noise=noise_arr[5])

        x = self.bottleneck1(x, s, noise1=noise_arr[6], noise2=noise_arr[7])
        x = self.bottleneck2(x, s, noise1=noise_arr[8], noise2=noise_arr[9])

        x = self.up1(x, s, noise1=noise_arr[10], noise2=noise_arr[11])
        x = torch.cat([narrow_like_(x2, x), x], dim=1)
        x = self.resblock3(x, s, noise1=noise_arr[12], noise2=noise_arr[13])

        x = self.up2(x, s, noise1=noise_arr[14], noise2=noise_arr[15])
        x = torch.cat([narrow_like_(x1, x), x], dim=1)
        x = self.resblock4(x, s, noise1=noise_arr[16], noise2=noise_arr[17])

        x = self.conv_out((x, s))

        return y + x
