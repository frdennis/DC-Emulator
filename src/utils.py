import torch
import numpy as np
import MAS_library as MASL
import Pk_library as PKL


def narrow_like_(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a

def narrow_by(a, c):
    """Narrow a by size c symmetrically on all edges.
    """
    ind = (slice(None),) * 2 + (slice(c, -c),) * (a.dim() - 2)
    return a[ind]


def chunk_upscale(input, style, model, device='cpu', nsplit=4, pad=3, scale_factor=2, verbose=True, noise_arr=[None,]*18):
    size = input.shape[1:]
    size = np.asarray(size)
    ndim = len(size)

    chunk_size = size // nsplit
    crop = np.broadcast_to(chunk_size, size.shape)

    reps = size // crop
    tot_reps = int(np.prod(reps))

    pad = np.broadcast_to(pad, (ndim, 2))

    tgt_size = crop[0]*scale_factor
    tgt_chunk = np.broadcast_to(tgt_size, size.shape)

    Ngrid_SR = size[0]*scale_factor
    SR_displacement = np.zeros([3, Ngrid_SR, Ngrid_SR, Ngrid_SR], dtype=np.float32)


    style = style.to(device)

    for idx in range(0, tot_reps):
        chunk = torch.tensor(cropfield(np.array(input), idx, reps, crop, pad))

        out = model(chunk.unsqueeze(0).to(device), style.unsqueeze(0), noise_arr=noise_arr).detach().numpy()
        out = narrow_like(out[0], tgt_size)

        ns = np.unravel_index(idx, reps) * tgt_chunk

        SR_displacement[:,ns[0]:ns[0]+tgt_size,ns[1]:ns[1]+tgt_size,ns[2]:ns[2]+tgt_size] = out

        if (idx+1)%nsplit == 0 and verbose:
            print("{}/{} done".format(idx+1, tot_reps), flush=True)
    
    return SR_displacement

def dis2pos(dis_field,boxsize,Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize #+ 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos<0] += boxsize
    pos[pos>boxsize] -= boxsize

    return pos

def mass_assignment(input, Ngrid, BoxSize=1000., MAS='PCS', verbose=True):
    input = dis2pos(np.array(input), 1000., Ngrid).reshape((3, Ngrid*Ngrid*Ngrid))

    input = np.moveaxis(input, -1, 0)

    delta = np.ones((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    MASL.MA(input, delta, BoxSize, MAS, verbose=verbose)
    delta /= np.mean(delta) # dtype=np.float64

    return delta



def power(input, BoxSize=1000., MAS='PCS', threads=16):
    Pk = PKL.Pk(np.array(input, dtype=np.float32), BoxSize, axis=1, threads=threads, MAS=MAS)
    k = Pk.k3D
    Pk = Pk.Pk[:,0]
    return k, Pk

from scipy.special import hyp2f1


def cropfield(field,idx,reps,crop,pad):
    start = np.unravel_index(idx, reps) * crop
    x = field.copy()
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1+d, mode='wrap')
    return x


def narrow_like(SR_box, tgt_grid):
    width = np.shape(SR_box)[1] - tgt_grid
    half_width = width // 2 
    begin, stop = half_width, tgt_grid + half_width
    return SR_box[:,begin:stop,begin:stop,begin:stop]


def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
    """
    OL = 1 - Om
    a = 1 / (1 + z)
    return a * hyp2f1(1, 1 / 3, 11 / 6, - OL * a ** 3 / Om) \
           / hyp2f1(1, 1 / 3, 11 / 6, - OL / Om)


def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1 + z)
    aa3 = OL * a ** 3 / Om
    return 1 - 6 / 11 * aa3 * hyp2f1(2, 4 / 3, 17 / 6, -aa3) \
           / hyp2f1(1, 1 / 3, 11 / 6, -aa3)


def H(z, Om=0.31):
    """Hubble in [h km/s/Mpc] for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1 + z)
    return 100 * np.sqrt(Om / a ** 3 + OL)


def dis(x, undo=False, a=0.3333, dis_std=6.0, **kwargs):
    z = 1 / a - 1
    dis_norm = dis_std * D(z)  # [Kpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm