import MAS_library as MASL
import Pk_library as PKL

import numpy as np
from .utils import dis2pos


def mass_assignment(input, Ngrid, BoxSize=1000., MAS='PCS', verbose=True):
    """Density field 1 + delta from a displacement field.

    Parameters
    ----------
    input : array, shape [3, Ngrid, Ngrid, Ngrid]
        Displacement field in Mpc/h, ordered on the Lagrangian lattice.
    Ngrid : int
        Particles per side; the density is painted on a grid of the same size.
    BoxSize : float
        Box size in Mpc/h.
    MAS : str
        Mass-assignment scheme ('NGP', 'CIC', 'TSC' or 'PCS').

    Returns
    -------
    np.ndarray, shape [Ngrid, Ngrid, Ngrid], float32, normalised to mean 1.
    """
    pos = dis2pos(np.asarray(input), BoxSize, Ngrid).reshape((3, Ngrid**3))

    # Pylians needs a C-contiguous float32 array of shape [Npart, 3]
    pos = np.ascontiguousarray(pos.T, dtype=np.float32)

    delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)
    delta /= np.mean(delta, dtype=np.float64)

    return delta


def power(input, BoxSize=1000., MAS='PCS', threads=16):
    """Monopole power spectrum of a density field; returns (k [h/Mpc], P(k) [(Mpc/h)^3])."""
    Pk = PKL.Pk(np.asarray(input, dtype=np.float32), BoxSize, axis=1, threads=threads, MAS=MAS)
    return Pk.k3D, Pk.Pk[:, 0]
