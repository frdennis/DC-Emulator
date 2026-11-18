# DC-Emulator

Code for the paper [**Separate Universe Super-Resolution Emulator**](https://arxiv.org/abs/2605.09004)
(Fremstad, Adamek & Mota 2026).

DC-Emulator is a generative model that super-resolves cosmological N-body
simulations, including simulations with **non-zero spatial curvature**. Given a
low-resolution (LR) displacement field, it generates a high-resolution (HR)
realisation with **2x the resolution per dimension (8x the particles)**,
conditioned on Ω_k, Ω_m, σ8, h and the scale factor. The model is trained as a
Wasserstein GAN, so the injected noise acts as stochastic small-scale structure,
and repeated runs give an ensemble of plausible HR realisations.

<!-- TODO: add a figure, e.g. LR vs SR density slices (Fig. 7 of the paper) -->

## Installation

```bash
git clone https://github.com/frdennis/DC-Emulator.git
cd DC-Emulator
pip install -r requirements.txt
```

### Model weights

The trained weights (~50 MB) are not stored in the repository. Download
`dc_emulator_weights.pt` from <!-- TODO: link to GitHub Release / Zenodo -->
and place it in the repository root. To keep them somewhere else, pass
`DCEmulator(state_path=...)`.

## Usage

```python
from src import DCEmulator

model = DCEmulator()   # uses a GPU if available

# Super-resolved displacement field [3, 2N, 2N, 2N] in Mpc/h
dis_SR = model.position("snapshot", "ics", sigma8=0.8, upscale=True)

# Super-resolved velocity field from a finite difference in time
vel_SR = model.velocity("snapshot", "ics", sigma8=0.8, dz=1e-5, upscale=True)
```

Inputs:

- `snapshot`: a low-resolution snapshot in Gadget format (CDM particles on an N³ grid).
- `ics`: the unperturbed initial conditions of the same simulation, which give the
  Lagrangian positions **q** (so that **x** = **q** + **Ψ**).
- `sigma8`: σ8 of the simulation. The other parameters are read from the snapshot header.

You can also upscale a displacement field you already have:

```python
dis_SR = model.upscale(dis_LR, a=0.5, Om_k=0.0, sigma8=0.8, Om_m=0.31, h=0.7)
```

The generator injects random noise to create small-scale structure, so each call
gives a new realisation. Pass `noise="fixed"` for a reproducible realisation (set
with `DCEmulator(seed=...)`). Velocities are computed by upscaling the snapshot at
z ± dz with the same noise and differencing the results (Sect. 4.2 of the paper).

See [`example.ipynb`](example.ipynb) for a full walkthrough, including density
fields and power spectra.

## Training data and range of validity

The model was trained on 32 pairs of separate-universe simulations in which
curvature enters only through the background expansion:

| | Low resolution (input) | High resolution (target) |
|---|---|---|
| Code | `gevolution` (particle-mesh, fixed grid) | `Gadget-4` (softening 50 h⁻¹ kpc) |
| Box size | 1000 h⁻¹ Mpc | 1000 h⁻¹ Mpc |
| Particles | 256³ | 512³ |

Each pair shares the same initial data at z = 63 (2LPT, with the LR initial
conditions obtained by downsampling), and each pair has a different random seed.
Snapshots were saved at 11 scale factors, a ∈ {0.25, 0.30, …, 0.75}.

The four cosmological parameters were sampled with a Latin hypercube over

| Parameter | Range |
|---|---|
| Ω_k | [−0.1, 0.1] |
| Ω_m | [0.29555, 0.32666] |
| σ8 | [0.76969, 0.85071] |
| h | [0.674, 0.768] |
| a | [0.25, 0.75] (z ≈ 0.33–3) |

with Ω_b = 0.049, n_s = 0.9624, T_CMB = 2.725 K and N_eff = 3.046 (massless
neutrinos) held fixed.

The model works chunk by chunk, so it can be applied to periodic boxes of any
size, but the **input should have the same mean particle spacing as the training
data, 1000/256 ≈ 3.9 h⁻¹ Mpc**. Results outside the parameter ranges above are
extrapolations; `DCEmulator` issues a warning if the inputs fall outside them.

## Performance

On the test set (see Sect. 4 of the paper):

- **Matter power spectrum:** within ~1% for k ≲ 0.2 h Mpc⁻¹, ~5% for
  k ≲ 1 h Mpc⁻¹, and ~10% up to the Nyquist wavenumber, with a residual
  suppression of small-scale power that is largest at low redshift.
- **Velocities:** the velocity divergence power spectrum is within ≲2% for
  k ≲ 0.7 h Mpc⁻¹ and ~10% at k ~ 1 h Mpc⁻¹.
- **Halos:** the SR halo mass function has ~20% (high z) to ~30% (low z) fewer
  low-mass halos, about 10% fewer around 5×10¹³ M⊙, and is consistent with
  the HR target above ~5×10¹⁴ M⊙. Halo central densities are underestimated by
  up to ~50%.

## Repository structure

```
src/
├── emulator.py   # DCEmulator: snapshot reading, upscaling, velocities
├── model.py      # UNetG generator network
├── layers.py     # Styled convolutions, noise injection, U-Net blocks
├── utils.py      # Chunked inference, normalisation, cosmology helpers
└── analysis.py   # Mass assignment and power spectra (requires Pylians)
example.ipynb     # Tutorial
```

## Citation

If you use this code, please cite

```bibtex
@article{Fremstad2026,
  author        = {Fremstad, Dennis and Adamek, Julian and Mota, David F.},
  title         = {Separate Universe Super-Resolution Emulator},
  year          = {2026},
  eprint        = {2605.09004},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO},
}
```

## Acknowledgements

The generator architecture is adapted from
[map2map](https://github.com/eelregit/map2map) and
[dl_halo](https://github.com/dschaurecker/dl_halo), with the noise injection
following Zhang et al. (2024). Snapshot reading, mass assignment and power
spectra use [Pylians](https://github.com/franciscovillaescusa/Pylians3).
