import numpy as np
import torch
from scipy.special import hyp2f1


# ---------------------------------------------------------------------------
# Cropping helpers
# ---------------------------------------------------------------------------

def narrow_like_(a, b):
    """Narrow torch tensor ``a`` to the spatial shape of ``b``.

    Try to be symmetric but cut more on the right for odd difference.
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a


def narrow_like(field, tgt_grid):
    """Crop a numpy field of shape [C, N, N, N] symmetrically to [C, tgt_grid, tgt_grid, tgt_grid]."""
    width = np.shape(field)[1] - tgt_grid
    begin = width // 2
    stop = begin + tgt_grid
    return field[:, begin:stop, begin:stop, begin:stop]


def cropfield(field, idx, reps, crop, pad):
    """Extract chunk number ``idx`` of a periodic field [C, N, N, N], padded by ``pad`` cells."""
    start = np.unravel_index(idx, reps) * crop
    x = field
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode='wrap')
    return x


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def chunk_upscale(input, style, model, nsplit=4, pad=3, scale_factor=2,
                  noise_arr=None, verbose=True):
    """Super-resolve a full periodic box by running the model chunk by chunk.

    The box is split into ``nsplit**3`` cubes, each padded periodically by
    ``pad`` cells, upscaled, and cropped back before being stitched together.

    Parameters
    ----------
    input : torch.Tensor or np.ndarray, shape [3, N, N, N]
        Normalised low-resolution displacement field.
    style : torch.Tensor, shape [5]
        Style vector ``[a, Om_k, sigma8, Om_m, h]``.
    model : torch.nn.Module
        The generator; inference runs on the device its parameters live on.
    noise_arr : list or None
        Fixed noise passed to the model (see ``model.make_noise``); None draws
        new noise for each chunk.

    Returns
    -------
    np.ndarray, shape [3, scale_factor*N, scale_factor*N, scale_factor*N], float32
    """
    device = next(model.parameters()).device

    field = np.asarray(input.cpu() if torch.is_tensor(input) else input)
    size = np.asarray(field.shape[1:])
    ndim = len(size)

    if np.any(size % nsplit):
        raise ValueError(f"Grid size {tuple(size)} is not divisible by nsplit={nsplit}")

    crop = size // nsplit
    reps = size // crop
    tot_reps = int(np.prod(reps))
    pad = np.broadcast_to(pad, (ndim, 2))

    tgt_size = crop[0] * scale_factor
    Ngrid_SR = size[0] * scale_factor
    output = np.zeros([field.shape[0], Ngrid_SR, Ngrid_SR, Ngrid_SR], dtype=np.float32)

    style = style.to(device).unsqueeze(0)

    with torch.no_grad():
        for idx in range(tot_reps):
            chunk = torch.from_numpy(cropfield(field, idx, reps, crop, pad)).unsqueeze(0).to(device)

            out = model(chunk, style, noise_arr=noise_arr)[0].cpu().numpy()
            out = narrow_like(out, tgt_size)

            n0, n1, n2 = np.array(np.unravel_index(idx, reps)) * tgt_size
            output[:, n0:n0+tgt_size, n1:n1+tgt_size, n2:n2+tgt_size] = out

            if verbose and ((idx + 1) % max(1, tot_reps // 4) == 0 or idx + 1 == tot_reps):
                print(f"{idx+1}/{tot_reps} chunks done", flush=True)

    return output


def dis2pos(dis_field, boxsize, Ng):
    """Convert a displacement field to particle positions.

    Assumes ``dis_field`` has shape [3, Ng, Ng, Ng] and is ordered on the
    Lagrangian lattice (particle ID order). Positions are wrapped periodically
    into [0, boxsize).
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos < 0] += boxsize
    pos[pos >= boxsize] -= boxsize

    return pos


# ---------------------------------------------------------------------------
# Cosmology and normalisation
# ---------------------------------------------------------------------------

def D(z, Om=0.31):
    """Linear growth function for flat LambdaCDM, normalised to 1 at z=0."""
    OL = 1 - Om
    a = 1 / (1 + z)
    return a * hyp2f1(1, 1 / 3, 11 / 6, - OL * a ** 3 / Om) \
           / hyp2f1(1, 1 / 3, 11 / 6, - OL / Om)


def f(z, Om=0.31):
    """Linear growth rate for flat LambdaCDM."""
    OL = 1 - Om
    a = 1 / (1 + z)
    aa3 = OL * a ** 3 / Om
    return 1 - 6 / 11 * aa3 * hyp2f1(2, 4 / 3, 17 / 6, -aa3) \
           / hyp2f1(1, 1 / 3, 11 / 6, -aa3)


def H(z, Om=0.31):
    """Hubble parameter in [h km/s/Mpc] for flat LambdaCDM."""
    OL = 1 - Om
    a = 1 / (1 + z)
    return 100 * np.sqrt(Om / a ** 3 + OL)


def dis(x, undo=False, a=0.3333, dis_std=6.0, **kwargs):
    """Normalise a displacement field in place (or undo the normalisation).

    The field is divided by ``dis_std * D(z)``. ``Om=0.31`` in ``D`` is fixed
    on purpose: this is the normalisation the network was trained with, and it
    is independent of the cosmology passed in the style vector.
    """
    z = 1 / a - 1
    dis_norm = dis_std * D(z)

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm
