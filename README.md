# Verification-Guided Nonlinear MPC

A nonlinear model predictive controller (MPC) for a kinematic bicycle vehicle whose safety
constraint comes from Hamilton–Jacobi (HJ) reachability, computed **online** while the
vehicle drives.

Each set has a time horizon and a spatial window. Two kinds of sets are computed:

- **Long-horizon sets.** A reach-avoid problem ("stay safe *until* the target band
  is reached") is solved over a window that moves with the vehicle. It considers the static
  obstacles only and is recomputed in a background thread on a fixed schedule.
- **Short-horizon sets.** These are computed more often over a short time window.
  They add the moving road users and use the latest long-horizon set as the target. To meet
  real-time deadlines, the short-horizon PDE is solved on a cropped grid:
  - in time, so the MPC never waits for a set;
  - in space, only on a box around the relevant part of the window.
- **MPC (acados).** It tracks progress towards the goal and requires the value of the
  short-horizon set in use to stay below `-SAFETY_MARGIN` at every stage of its horizon.

The HJ sets are computed with [pyspect](https://github.com/KTH-SML/pyspect) on top of
[hj_reachability](https://github.com/StanfordASL/hj_reachability) (JAX). The MPC is
generated with [acados](https://github.com/acados/acados).

## Repository structure

| File | Content |
| --- | --- |
| `runtime.py` | Entry point. Starts the verifier, controller and animation as separate processes that share a registry |
| `config.py` | All parameters: scenario, HJ grid and schedule, cropping, MPC weights and limits |
| `verifier.py` | Online HJ reachability: long- and short-horizon sets, schedule, spatial/time cropping |
| `controller.py` | Nonlinear MPC (acados) with the HJ safety constraint |
| `obstacles.py` | Static and moving obstacles (shared by the verifier and the animation) |
| `animation.py` | Closed-loop animation, timeline plot, and run data saved for the figures |
| `urban_scene.py` | Top-view drawing of the urban road scenario |
| `log.py` | Console logging shared by all processes |
| `plots/` | Scripts that generate the figures of the paper (see [Figures](#figures)) |
| `TTR.ipynb`, `map.ipynb`, `dynamic.ipynb` | Exploration notebooks (scenario figures, time-to-reach maps, set computation) |

## Installation

**Requirements:** Linux, **Python 3.13 or newer** (tested with Python 3.14), and
optionally an NVIDIA GPU. Python 3.12 does not work: the pinned version of `pyspect` uses
`warnings.deprecated`, which only exists from Python 3.13.

Steps 1 and 2 install all the Python packages. Step 3 is only needed for a GPU. Step 4
(acados) is a C library that pip cannot install, so it is set up separately. Step 5 is
only needed for two of the figure scripts.

### 1. Clone the repository and create a virtual environment

```bash
git clone https://github.com/KTH-SML/Verification-Guided-Nonlinear-MPC.git
cd Verification-Guided-Nonlinear-MPC
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Run `source .venv/bin/activate` again in every new terminal before using the code.

### 2. Install the Python packages

```bash
pip install -r requirements.txt
```

This installs all the Python dependencies at the versions used for the paper, including
`pyspect` from GitHub. JAX is installed in its CPU version: everything runs without a
GPU, but the reachable sets are computed much more slowly, so many of them are ready
too late (see `published late` in the summary printed at the end of a run).

### 3. (Optional) GPU support

Install the CUDA build of the **same** JAX version (`==0.11.1`, otherwise pip upgrades
JAX). Use `cuda13` for recent NVIDIA drivers, `cuda12` for older ones. CUDA does not
need to be installed on the system: pip installs the CUDA libraries in the virtual
environment.

```bash
pip install "jax[cuda13]==0.11.1"
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

The second command must print `gpu [CudaDevice(id=0)]`. If it prints `cpu`, JAX does not
see the GPU. If it fails with `No visible GPU devices`, check `nvidia-smi`: the message
`Driver/library version mismatch` means that the NVIDIA driver was updated while the
machine was running, and the machine must be rebooted.

### 4. acados

acados is a C library: clone it with its submodules and build it once with CMake (needs
`git`, `cmake`, `gcc`, `g++` and `make`; on Fedora `sudo dnf install cmake gcc gcc-c++ make`,
on Ubuntu `sudo apt install cmake build-essential`). The commands below install it in
`~/acados`; see the [acados installation guide](https://docs.acados.org/installation/) for
other options.

```bash
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake ..
make install -j$(nproc)
ls ~/acados/lib                     # must contain libacados.so
```

Set two environment variables with **absolute** paths. Adding them to `~/.bashrc` sets them
in every new terminal:

```bash
echo 'export ACADOS_SOURCE_DIR=$HOME/acados' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/acados/lib' >> ~/.bashrc
source ~/.bashrc
```

Install the Python interface of acados in the virtual environment, and check it:

```bash
cd ~/Verification-Guided-Nonlinear-MPC   # the folder of this repository
source .venv/bin/activate
pip install -e ~/acados/interfaces/acados_template
python -c "from acados_template.utils import get_tera; print(get_tera())"
```

The last command downloads the `t_renderer` tool of acados (if it is not there yet) and
prints its path, `~/acados/bin/t_renderer`.

### 5. (Optional) Export of the 3D figures

```bash
pip install kaleido
```

Only needed by `plots/plot_long_phi.py` and `plots/plot_short_phi.py` to save PDF/PNG
files. Without it, they still save an interactive HTML file.

## Running a simulation

```bash
python3 runtime.py                       # urban road scenario (default)
SCENARIO=corridor python3 runtime.py     # small-scale corridor scenario
```

Two scenarios are defined in `config.py`:

- `urban_road`: a full-size car on a 120 m two-lane road. Static blockages force lane
  changes, and road users cross the road: two cars, a pedestrian and a tram.
- `corridor`: a 1:10 scale vehicle in a corridor with a slalom of blocks and one moving box.

The main options are in `config.py`. Among them:

- `HJ_CROP_METHOD`: spatial crop of the short-horizon PDE (`'none'`, `'reach'` or
  `'zero_level'`)
- `HJ_PREDICTION`, `HJ_DELAY`, `HJ_SHORT_PERIOD`: short-horizon schedule
- `HJ_HORIZON`, `LONG_UPDATE`, `LONG_LEAD`: long-horizon schedule
- `SAFETY_V_MODE`: value function used in the MPC constraint

The comments in `config.py` give the timing requirements these values must satisfy.

A run writes:

| Output | Content |
| --- | --- |
| `sets/` | Value functions of the short-horizon sets |
| `vf_debug/`, `vf_debug.gif` | Animation (vehicle, sets in use, value along the trajectory): frames and GIF |
| `timeline.png` | When each set was computed and used |
| `results/run_data.pkl` | Trajectory, timelines and value function, used by the figure scripts |
| `results/crop_<method>.csv` | Per-set statistics of the short-horizon PDE (grid size, compute time) |
| `codegen_ocp_*/`, `acados_ocp.json` | Solver code generated by acados |

None of these are tracked by git. Each run overwrites the outputs of the previous one (`crop_<method>.csv`: the file of the same crop method).

## Figures

The figure scripts are in `plots/`. Run them from the repository root. They write to
`results/paper/`.

| Script | Figure | Input |
| --- | --- | --- |
| `plots/plot_paper.py` | Value `V(z(t))` along the closed-loop trajectory, and the schedule of the long/short-horizon sets | `results/run_data.pkl` |
| `plots/plot_snapshots.py` | Scenario and closed-loop snapshots with the sets in use | `results/run_data.pkl`, `sets/` |
| `plots/plot_waiting.py` | Vehicle waiting in front of the tram: how the short-horizon set changes | `results/run_data.pkl`, `sets/` |
| `plots/plot_intervals.py` | Schematic of the computation, prediction and use times of the sets | none |
| `plots/plot_long_phi.py` | 3D zero level set of the first long-horizon set in (x, y, t) | computed by the script |
| `plots/plot_short_phi.py` | 3D zero level set of a short-horizon set with the moving road users | computed by the script |

Example:

```bash
python3 runtime.py                                # run the simulation first
python3 plots/plot_paper.py
python3 plots/plot_snapshots.py --times 6 14
python3 plots/plot_waiting.py
python3 plots/plot_intervals.py
python3 plots/plot_long_phi.py                  # needs kaleido for PDF/PNG
python3 plots/plot_short_phi.py --k 1
```

Each script lists its options with `--help`.
