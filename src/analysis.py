import MAS_library as MASL
import Pk_library as PKL

import numpy as np
from .utils import dis2pos


def mass_assignment(input, Ngrid, BoxSize=1000., MAS='PCS', verbose=True):
    input = dis2pos(np.array(input), BoxSize, Ngrid).reshape((3, Ngrid*Ngrid*Ngrid))

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
