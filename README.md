# DC-Emulator

A deep-learning emulator that super-resolves cosmological N-body simulations.
Given a low-resolution snapshot, it predicts the displacement and velocity fields
of a simulation with **2x the resolution per dimension (8x the particles)**,
conditioned on the cosmological parameters.

The generator is a style-conditioned 3D U-Net (StyleGAN2-like modulated
convolutions with noise injection), trained adversarially. The cosmology enters
through the style vector `[a, Ω_k, σ8, Ω_m, h]`.

<!-- TODO: add a figure, e.g. LR vs SR density slices and power spectra from example.ipynb -->

## Installation

```bash
git clone https://github.com/frdennis/DC-Emulator.git
cd DC-Emulator
pip install -r requirements.txt
```

### Model weights

The trained weights (~50 MB) are not stored in the repository.
Download `state.pt` from <!-- TODO: link to GitHub Release / Zenodo --> and place it in `src/`.

## Usage

```python
from src.DCGAN import DCGAN

model = DCGAN(state_path="src/state.pt")   # uses a GPU if available

# Super-resolved displacement field [3, 2N, 2N, 2N] in Mpc/h
dis_SR = model.position("snapshot", "ics", sigma8=0.778, upscale=True)

# Super-resolved velocity field from a finite difference in time
vel_SR = model.velocity("snapshot", "ics", sigma8=0.778, dz=1e-5, upscale=True)
```

Inputs:

- `snapshot`: a low-resolution Gadget snapshot of CDM particles on an N³ grid.
- `ics`: the unperturbed initial conditions of the same simulation, which give the
  Lagrangian positions **q** (so that **x** = **q** + **ψ**).
- `sigma8`: σ8 of the simulation. The other parameters are read from the snapshot header.

You can also upscale a displacement field you already have:

```python
dis_SR = model.upscale(dis_LR, a=0.5, Om_k=0.0, sigma8=0.8, Om_m=0.31, h=0.68)
```

The generator injects random noise to create small-scale structure. Pass
`noise="fixed"` for a reproducible realisation (set with `DCGAN(..., seed=...)`).

See [`example.ipynb`](example.ipynb) for a full walkthrough, including density
fields and power spectra.

## Training data and validity

<!-- TODO: describe the simulations the model was trained on (code, box size,
     resolution, redshifts) and the parameter ranges of a, Ω_k, σ8, Ω_m and h.
     The model should not be expected to work outside these ranges. -->

## Repository structure

```
src/
├── DCGAN.py      # DCGAN class: snapshot reading, upscaling, velocities
├── model.py      # UNetG generator network
├── layers.py     # Styled convolutions, noise injection, U-Net blocks
├── utils.py      # Chunked inference, normalisation, cosmology helpers
└── analysis.py   # Mass assignment and power spectra (requires Pylians)
example.ipynb     # Tutorial
```

## Citation

<!-- TODO: add the paper / BibTeX entry once available -->

## Acknowledgements

The generator architecture is adapted from
[map2map](https://github.com/eelregit/map2map) and
[dl_halo](https://github.com/dschaurecker/dl_halo).
Snapshot reading, mass assignment and power spectra use
[Pylians](https://github.com/franciscovillaescusa/Pylians3).
