import numpy as np
import torch
import scipy.integrate as integrate
import readgadget

from .model import UNetG, make_noise
from .utils import chunk_upscale, dis


STYLE_PARAMS = ("a", "Om_k", "sigma8", "Om_m", "h")


class DCGAN:
    """Super-resolution emulator for N-body displacement and velocity fields.

    Takes a low-resolution (LR) simulation snapshot and predicts the
    displacement field of a simulation with 2x the resolution per dimension
    (8x the particles), conditioned on cosmology.

    Parameters
    ----------
    state_path : str
        Path to the trained weights (a ``.pt`` file containing a ``'model'`` state dict).
    device : str or torch.device, optional
        Where to run the network, e.g. ``"cpu"`` or ``"cuda"``. Defaults to
        ``"cuda"`` if available, otherwise ``"cpu"``.
    seed : int, optional
        Seed for the fixed noise realisation used by ``noise="fixed"`` and by
        ``velocity``. If None, a random seed is used.
    nsplit : int
        The box is processed in ``nsplit**3`` chunks to limit memory use.
    pad : int
        Number of periodic padding cells on each side of a chunk.
    verbose : bool
        Print progress while upscaling.
    """

    def __init__(self, state_path, device=None, seed=None, nsplit=4, pad=3, verbose=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.seed = seed
        self.nsplit = nsplit
        self.pad = pad
        self.verbose = verbose

        state = torch.load(state_path, map_location="cpu")
        self.model = UNetG(in_chan=3, out_chan=3, style_size=len(STYLE_PARAMS))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()

        self._fixed_noise = {}   # chunk size -> fixed noise realisation

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def fixed_noise(self, Ngrid):
        """Return the fixed noise realisation for an input box of ``Ngrid**3`` cells."""
        chunk_size = Ngrid // self.nsplit + 2 * self.pad
        if chunk_size not in self._fixed_noise:
            self._fixed_noise[chunk_size] = make_noise(chunk_size, seed=self.seed)
        return self._fixed_noise[chunk_size]

    def upscale(self, x, s=None, noise=None, *,
                a=None, z=None, Om_k=None, sigma8=None, Om_m=None, h=None):
        """Upscale a low-resolution displacement field by 2x per dimension.

        The cosmology is given either as a style vector ``s`` or with keyword
        arguments (``z`` may be given instead of ``a``).

        Parameters
        ----------
        x : np.ndarray, shape [3, N, N, N]
            Low-resolution displacement field in Mpc/h. Not modified.
        s : array-like, shape [5], optional
            Style vector ``[a, Om_k, sigma8, Om_m, h]``.
        noise : None, "fixed" or list, optional
            None draws new random noise (a new realisation every call).
            ``"fixed"`` uses the model's fixed realisation (see ``seed``), so
            repeated calls are deterministic. A list from ``model.make_noise``
            uses that realisation.

        Returns
        -------
        np.ndarray, shape [3, 2N, 2N, 2N]
            Predicted high-resolution displacement field in Mpc/h.
        """
        if z is not None and a is None:
            a = 1.0 / (1.0 + z)

        if s is None:
            values = dict(a=a, Om_k=Om_k, sigma8=sigma8, Om_m=Om_m, h=h)
            missing = [k for k, v in values.items() if v is None]
            if missing:
                raise ValueError(f"Missing parameter(s): {', '.join(missing)}")
            s = [values[k] for k in STYLE_PARAMS]

        s = torch.as_tensor(np.asarray(s, dtype=np.float32).ravel())
        if s.numel() != len(STYLE_PARAMS):
            raise ValueError(f"Style vector must contain exactly {len(STYLE_PARAMS)} "
                             f"parameters: {list(STYLE_PARAMS)}")

        if isinstance(noise, str):
            if noise != "fixed":
                raise ValueError(f"Unknown noise option {noise!r}")
            noise = self.fixed_noise(x.shape[1])

        x = np.array(x, dtype=np.float32)   # copy, so the caller's array is not modified
        a = float(s[0])

        dis(x, a=a)
        y = chunk_upscale(x, s, self.model, nsplit=self.nsplit, pad=self.pad,
                          noise_arr=noise, verbose=self.verbose)
        dis(y, a=a, undo=True)

        return y

    # ------------------------------------------------------------------
    # Snapshot interface
    # ------------------------------------------------------------------

    def position(self, snapshot, snapshot_ic, sigma8, upscale=True, noise=None):
        """Displacement field of a Gadget snapshot, optionally super-resolved.

        Parameters
        ----------
        snapshot : str
            Path to the low-resolution Gadget snapshot (CDM particles, type 1).
        snapshot_ic : str
            Unperturbed initial conditions (particles on the Lagrangian lattice),
            used to obtain the Lagrangian positions q.
        sigma8 : float
            sigma_8 of the simulation (not stored in the Gadget header).
        upscale : bool
            If False, return the low-resolution field from the snapshot.
        noise : None, "fixed" or list
            See ``upscale``.

        Returns
        -------
        np.ndarray, shape [3, N, N, N] (or [3, 2N, 2N, 2N] if ``upscale``)
            Displacement field psi = x - q in Mpc/h, ordered on the Lagrangian lattice.
        """
        style = self._style_from_header(snapshot, sigma8)
        dis_lr = self.make_lag_dz(snapshot, snapshot_ic, dz=0.0)

        if not upscale:
            return dis_lr
        return self.upscale(dis_lr, style, noise=noise)

    def velocity(self, snapshot, snapshot_ic, sigma8, dz=None, da=None, upscale=True):
        """Velocity field v = (a/h) dx/dt from a central finite difference in time.

        The particles are drifted to t +/- dt using their snapshot velocities,
        each state is upscaled, and the difference gives the velocity. The same
        fixed noise realisation is used for both states, so the difference
        reflects the change in the field rather than a change in noise.

        Parameters
        ----------
        snapshot, snapshot_ic, sigma8 :
            As in ``position``.
        dz, da : float
            Half-width of the time step in redshift or in scale factor.
            Exactly one must be given.
        upscale : bool
            If False, return the low-resolution velocity field.

        Returns
        -------
        np.ndarray, shape [3, N, N, N] (or [3, 2N, 2N, 2N] if ``upscale``)
        """
        header = readgadget.header(snapshot)
        a, z = header.time, header.redshift
        Om_m, Om_l, h = header.omega_m, header.omega_l, header.hubble
        Om_k = 1. - (Om_m + Om_l)

        # "p" is the later time (larger a), "m" the earlier time
        if da is not None and dz is None:
            a_p, a_m = a + da, a - da
        elif dz is not None and da is None:
            a_p, a_m = 1 / (1 + z - dz), 1 / (1 + z + dz)
        else:
            raise ValueError("Give exactly one of dz or da")
        dz_p = 1 / a_p - 1 - z
        dz_m = 1 / a_m - 1 - z

        input_p = self.make_lag_dz(snapshot, snapshot_ic, dz_p)
        input_m = self.make_lag_dz(snapshot, snapshot_ic, dz_m)
        dt = self.t(Om_k, Om_m, Om_l, h, a_m, a_p)

        if upscale:
            style = [a, Om_k, sigma8, Om_m, h]
            style_p = [a_p] + style[1:]
            style_m = [a_m] + style[1:]
            input_p = self.upscale(input_p, style_p, noise="fixed")
            input_m = self.upscale(input_m, style_m, noise="fixed")

        return (input_p - input_m) / dt * a / h

    # ------------------------------------------------------------------
    # Snapshot reading
    # ------------------------------------------------------------------

    @staticmethod
    def _style_from_header(snapshot, sigma8):
        header = readgadget.header(snapshot)
        Om_m, Om_l = header.omega_m, header.omega_l
        return [header.time, 1. - (Om_m + Om_l), sigma8, Om_m, header.hubble]

    def readsnap(self, file):
        """Read CDM positions [Mpc/h], velocities [km/s] and IDs (starting at 0) from a Gadget file."""
        ptype = [1]

        pos = np.array(readgadget.read_block(file, "POS ", ptype)) / 1e3   # kpc/h -> Mpc/h
        vel = np.array(readgadget.read_block(file, "VEL ", ptype))         # peculiar velocity
        ids = np.array(readgadget.read_block(file, "ID  ", ptype))

        ids -= ids.min()
        header = readgadget.header(file)

        return pos, vel, ids, header

    def order(self, arr, ids):
        """Sort particle data by ID, dropping IDs that are not present."""
        # *8 since the IC IDs are taken from high-resolution ICs with 8x more particles
        _arr = np.full((arr.shape[0] * 8, 3), np.nan)
        _arr[ids] = arr
        return _arr[~np.isnan(_arr).any(axis=1)]

    def make_lag_dz(self, file, file_ic, dz):
        """Displacement field [3, N, N, N] in Mpc/h, with particles drifted by ``dz`` in redshift.

        ``dz=0`` gives the displacement at the snapshot time. Positive ``dz``
        drifts the particles backwards in time (to higher redshift).
        """
        pos, vel, ids, header = self.readsnap(file)
        pos_ic, _, ids_ic, header_ic = self.readsnap(file_ic)

        Npart = pos.shape[0]
        Ngrid = int(np.rint(Npart**(1/3)))
        BoxSize = header.boxsize / 1e3

        a = header.time
        Om0 = header.omega_m
        Ol = header.omega_l
        Omk = 1. - Om0 - Ol

        # Drift dx = v dt, with dt written in terms of dz. The sqrt(a) is needed
        # because readgadget multiplies the velocities by sqrt(a) in read_block.
        dx_u = -10.0 * dz * np.sqrt(a) / np.sqrt(
            (Om0 / a) / a
            + Ol * a
            + Omk / a
        )

        assert Npart == header_ic.nall[1], "Snapshot and IC files have different particle numbers"

        pos = self.order(pos, ids)
        vel = self.order(vel, ids)
        q = self.order(pos_ic, ids_ic)

        # Snap the Lagrangian positions to the grid
        q = (q + BoxSize) % BoxSize
        indxs = (np.rint(q / BoxSize * Ngrid) % Ngrid).astype(int)
        q = indxs / Ngrid * BoxSize

        disp = (pos + vel * dx_u / 1e3) - q

        dis_field = np.full([Ngrid, Ngrid, Ngrid, 3], np.nan)
        dis_field[indxs[:, 0], indxs[:, 1], indxs[:, 2]] = disp
        dis_field -= np.rint(dis_field / BoxSize) * BoxSize   # periodic wrap

        assert not np.isnan(dis_field).any(), "Some grid cells received no particle"

        return np.moveaxis(dis_field, -1, 0)

    # ------------------------------------------------------------------
    # Cosmology
    # ------------------------------------------------------------------

    @staticmethod
    def H(a, Om_k, Om_m, Om_l, h):
        """Hubble parameter in km/s/Mpc (radiation neglected)."""
        return 100 * h * np.sqrt(Om_m * a**-3 + Om_k * a**-2 + Om_l)

    def t(self, Om_k, Om_m, Om_l, h, a1, a2):
        """Time between scale factors a1 and a2, int da / (a H)."""
        return integrate.quad(
            lambda x: 1 / (x * self.H(x, Om_k, Om_m, Om_l, h)), a1, a2
        )[0]
