# cuda_muons: GPU muon propagation

Fast simulation of muons through magnetised iron (the SHiP muon shield) on the GPU. Each muon is propagated
in fixed steps. In every step in matter, the energy loss and transverse momentum kick are sampled from
2D histograms built from Geant4 single-step simulations, and the trajectory is integrated through the
magnetic field with a Runge-Kutta step. It is used standalone (`cuda_muons.py`) or from
[MuonsAndMatter](https://github.com/lfpc/MuonsAndMatter) (`muons_and_matter/cuda_muons_ship.py`, same detector
as the Geant4 simulation).

## Layout

| Path | Content |
|---|---|
| `cuda_muons.py` | `set_environment`, `propagate_muons_with_cuda`, `run_from_params` and the command-line script |
| `faster_muons_torch/` | CUDA extension (kernels for uniform fields and for field maps, PyTorch bindings) |
| `utils_cuda_muons/get_geometry.py` | Magnet blocks (ARB8) and cavern from the magnet parameters |
| `utils_cuda_muons/get_magnetic_field.py` | Uniform fields per block, FEM field maps (snoopy) and field-map files |
| `utils_cuda_muons/collect_geant_data.py`, `unify_data.py`, `build_histograms.py` | Building the material histograms |
| `data/alias_histograms_G4_Fe.pkl`, `data/alias_histograms_G4_CONCRETE.pkl` | Histograms for iron and concrete |
| `geant4/` | Minimal Geant4 module, only needed to build histograms outside MuonsAndMatter |
| `experiments/` | Benchmarks and comparisons with Geant4 |

## Installation

Needs a CUDA GPU, PyTorch with CUDA, and an `nvcc` that supports your host compiler (e.g. CUDA 12.4 needs
gcc ≤ 13). On the UZH physik cluster, use the MuonsAndMatter container (`bash shell_container.sh` from the
MuonsAndMatter root).

Build and install the extension (from this folder, once, and again after any change in `faster_muons_torch/`):

```bash
bash install_cuda.sh      # pip3 install --no-build-isolation --force-reinstall --user ./faster_muons_torch
```

Inside the container, clear these variables first if the build fails:

```bash
export PYTHONNOUSERSITE=0
aux_ld_preload="$LD_PRELOAD"
export LD_PRELOAD=""
source install_cuda.sh
export LD_PRELOAD="$aux_ld_preload"
unset aux_ld_preload
export PYTHONNOUSERSITE=1
```

## Running

From this folder:

```bash
python3 cuda_muons.py -params data/tokanut_v6.txt -muons ../data/muons/full_sample_after_target.h5 -sens_plane 82
```

| Option | Meaning |
|---|---|
| `-params` | Text file with the magnet parameters (15 per magnet, see the MuonsAndMatter README) |
| `-muons` / `-n_muons` | Input muons (`.npy`, `.pkl` or `.h5`) / maximum number (0 = all) |
| `-sens_plane` | z positions (m) of the sensitive planes |
| `-field_mode {uniform,read_file,simulate}` | Uniform field per block (default), field map from `-field_file`, or FEM map simulated with snoopy (saved to `-field_file` if given) |
| `-field_file` | Field map file |
| `-remove_cavern` | No cavern (no concrete) |
| `--n_steps` | Maximum number of steps per plane (default 5000) |
| `--h` | Folder with the histograms (default `data/`) |
| `--save_dir` | Save the output (pickle) to this path; nothing is saved by default |
| `--gpu` | GPU index |
| `-plot` | Plot input and output distributions |

Input muons have columns `[px, py, pz, x, y, z, pdg_id, weight]` (GeV/c, m; `pdg_id` ±13 or charge ±1;
`weight` optional), or `.h5` datasets `px, py, pz, x, y, z, pdg, weight`. The output is a dict with
`px, py, pz, x, y, z, pdg_id` (and `weight`) of the muons that hit all the sensitive planes; with
`return_all=True`, every muon in input order, the ones that missed a plane with zero momentum.

From Python, to propagate several times with the same geometry and field, build the GPU data once:

```python
env = set_environment(corners, cavern, material_histograms, magnetic_field)   # field: (N_blocks, 3) tensor or field-map dict
pos, mom = propagate_muons_with_cuda(positions, momenta, charges, env, sensitive_plane_z, n_steps, step_length, use_symmetry, seed)
```

## How the propagation works

- **Materials**: iron inside the magnet blocks, concrete outside the cavern walls, air elsewhere (no
  scattering in air). Only the quadrant x, y ≥ 0 of the geometry is stored; the lookup uses `(|x|, |y|, z)`.
- **Steps**: 2 cm (the step length of the histograms). 20 cm (`BIG_STEP`) only inside the cavern, after the
  last magnet (and at least z = 30 m), after the end of the field map, and more than 20 cm before the next
  sensitive plane.
- **Matter**: in each step in iron or concrete, `log(Δp/p)` and `log(p_T/p)` are sampled from the 2D histogram
  of the muon's momentum bin (alias method), and applied in a random azimuthal direction.
- **Field**: uniform mode, the field of the block the muon is in (zero in air); field-map mode, the nearest grid
  point of the map (zero outside it). The field of the other quadrants is mirrored: `Bx` changes sign where
  `x·y < 0`, `Bz` where `y < 0`. The trajectory is integrated with RK4, using the field at the start of the step
  and an average of the start and end fields for the later stages (`rk4_step_cached`).
- **Stopping**: a muon stops when it reaches the plane, when p < 0.18 GeV, when pz < 0, when |x| or |y| > 10 m,
  or after `n_steps` steps.
- **Random numbers**: each muon has its own random stream (seed, muon index); plane `i` uses `seed + i`.

Field-map files: HDF5 with `B` (N × 3, T) and `d_space` (3 × 3, rows `[min, max, step]` in cm for x, y, z), only
the quadrant x, y ≥ 0, points ordered y slowest, then x, then z fastest. See the MuonsAndMatter README for details.

## Material histograms

`data/` has the histograms for iron (`G4_Fe`) and concrete (`G4_CONCRETE`). They have 95 log-spaced momentum
bins from 0.18 to 400 GeV; muons above 400 GeV use the last bin (a rough approximation, since the scattering
and energy-loss fractions are taken at 400 GeV).

To build them for another material (Geant4 name), from this folder:

```bash
python3 utils_cuda_muons/collect_geant_data.py --material G4_Fe --num_sims 5000000 --cores 64   # Geant4 single steps -> data/muon_data_energy_loss_sens_<material><tag>.h5
python3 utils_cuda_muons/unify_data.py --material G4_Fe     # only if the data was collected in several files (--tag)
python3 utils_cuda_muons/build_histograms.py --material G4_Fe --alias    # -> data/alias_histograms_G4_Fe.pkl
```

`collect_geant_data.py` needs the Geant4 module `muon_slabs`: inside MuonsAndMatter, the one from
`build_cpp.sh` (on `PYTHONPATH` via `set_env.sh`); standalone, build `geant4/` with `source build_geant4.sh`.

**The momentum range is hardcoded in the CUDA code**: 0.18 and 400 GeV in `get_first_bin`
(`faster_muons_torch/common.cuh`) and in both launchers (`cuda_muons_field_map.cu`, `cuda_muons_uniform_field.cu`).
If you change `--initial_momenta` (min, max), change these values too and reinstall the extension, otherwise
muons are silently given the wrong momentum bin. The number of bins is read from the histograms, and the bins
must be log-spaced (do not use `--linearspace`). The cut `kill_at = 0.18` GeV in `cuda_muons.py` should match the
lower limit.

## Experiments

- `experiments/compare_geant4_and_histos.py`: compares cuda_muons with Geant4 for muons through a block of
  material (`--material`, `--initial_momenta`, `--mag_field`).
- `experiments/benchmark_time.py`: run time of cuda_muons against Geant4 as a function of the number of muons.
