# null-adapter

Range / conditional-residual decomposition of rare-sensor signals (e.g. hyperspectral)
through a known physical operator `A` (camera sensitivity × illuminant).

- `x_R = A⁺A x` — fully determined by RGB
- `x_N = (I − A⁺A) x` — invisible to RGB (`A x_N = 0`)
- `r = x − E[x | c]` — the part RGB *truly* cannot know (conditional residual)

Idea: keep the RGB model frozen for the range side, learn only the conditional
residual from few labels.

## Layout

```
Dockerfile, build_docker.sh, run_docker.sh   # Python 3.11, torch 2.4.1 (CPU), colour-science 0.4.4
toy1/                                       # Phase 1: synthetic spectra with closed-form ground truth
  configs/default.yaml   fixed hyperparameters & seeds (linear Gaussian prior)
  configs/mog.yaml       non-linear prior (mixture of Gaussians), inherits default.yaml
  src/operator.py        A, projectors, closed-form E[x|c], metamer sampler (generic in A)
  src/data.py            Gaussian spectral prior, tasks, metameric pairs
  src/encoders.py        phi (RGB "foundation model"), g, predictor, M5-MLP, RND-SCI generator, M6b
  src/methods.py         M0-M6b + reconstruction baselines (shared linear-probe head)
  src/baselines.py       HSI-Adapter-lite (end-to-end)
  src/run.py             experiment loop -> CSV
  tests/                 unit tests (projectors, metamers, posterior vs MC, label independence)
phase2/                                     # LIB-HSI, frozen DINOv3, per-pixel linear head (src/run.py)
phase3/                                     # same inputs, shared spatial head with zero-init gate (src/run.py)
phase4/                                     # whole-image labels, Phase 2 + 3 methods (src/run_linear.py, run_spatial.py)
polar/                                      # polarisation: SimCMF vs Null-Adapter on ZJU-RGBP / RGBP-Glass (SAM-B + LoRA)
```

Data are not in the repository: LIB-HSI (CC BY-NC 4.0, `phase2/scripts/fetch_libhsi.py` -> `phase2/data/full/LIB-HSI`)
and the SFU reflectance / sensor files (`toy1/data/raw/sfu/`) have to be fetched separately.

## Run

```bash
./build_docker.sh gpu        # or: ./build_docker.sh cpu   (VARIANT=cpu ./run_docker.sh ...)
./run_docker.sh bash -c "cd toy1 && python -m pytest -q tests"
./run_docker.sh bash -c "cd toy1 && python -m src.run --out results/main"
# non-linear prior: --config configs/mog.yaml --out results/mog
# LIB-HSI (VARIANT=phase2): phase2/scripts/prepare.py + features.py, then phase2/src/run.py,
# phase3/scripts/pipeline.sh, phase4/scripts/pipeline.sh
```

GPU image: torch 2.4.1 + CUDA 12.1 (`--gpus all`; seed s runs on cuda:(s % n_gpu)). Full Toy 1 run is about 3 min.
Proxy variables are forwarded to the build and to the container.

## Relation to existing methods

| here | reference | what is kept |
|---|---|---|
| `B_RND`, `B_RNDres` | RND-SCI (hustvl/RND-SCI, `model/RND.py`) | `x̂ = A⁺c + P_N h(c)`: the network only fills the null space, so `A x̂ = c` |
| `B_HSIA` | HSI-Adapter (robot-learning-freiburg/HSI-Adapter) | per-pixel spectral module → frozen RGB backbone, plus an HSI adapter injected through a zero-initialized gate, trained end to end on labels |
| `M1`, `M2` | HSI-Adapter / SimMAT family | learn from all channels (`x`) with or without frozen RGB features |

## Polarisation (polar/): comparison with SimCMF

SAM-B + LoRA from the SimCMF code (github.com/mt-cly/SimCMF @ 0cdc5c1; no license file, so it is not vendored:
clone it to `../ext/SimCMF` or set `SIMCMF_ROOT`, and put `sam_vit_b_01ec64.pth` in its `checkpoint/sam/`).
Only `image_encoder.projector` changes between methods; the recipe and the click/instance-IoU evaluation are SimCMF's.

| method | input to SAM's patch embedding |
|---|---|
| `simcmf` | SimCMF projector on [I, AoLP, DoLP] (9 ch, SimCMF normalisation) |
| `rgb` | pretrained embedding of the intensity image c only |
| `fuse_p` | `E(c) + sum_i w_i E_i(p_i)`, p = normalised Stokes s1/S0, s2/S0 per colour (6 ch), w zero-initialised |
| `null_p` | same module with r = p − m(c); m ≈ E[p | c] a CNN fitted label-free on all train RGB–P pairs |

Data: the archives linked in the SimCMF README (`zju-rgbp`, `pgsnet`) unpacked under `polar/data/raw/`, then
`python -m polar.scripts.prepare zju polar/data/raw/zju/zju-rgbp` (and `pgsnet …/pgsnet/pgsnet`).
Environment (RTX 5080 / sm_120 needs torch ≥ 2.7): `uv venv .venv && uv pip install torch==2.7.1 torchvision==0.22.1
--index-url https://download.pytorch.org/whl/cu128 && uv pip install "numpy<2" monai==1.3.1 einops opencv-python-headless
pillow tensorboardX tqdm pandas scipy pyyaml pytest`. Global attention runs through SDPA (exact reformulation of the
decomposed relative-position bias, `polar/src/bridge.py`, checked in `polar/tests/test_bridge.py`), so batch 4 fits 16 GB.

Docker (same versions; needs Docker Engine + NVIDIA Container Toolkit and a host driver ≥ 570):

```bash
./build_docker.sh polar                        # null-adapter:polar; host CA bundle passed as a build secret for the proxy
VARIANT=polar ./run_docker.sh python -m pytest -q polar/tests
VARIANT=polar ./run_docker.sh polar/scripts/queue.sh   # all runs, priority order, restartable
VARIANT=polar ./run_docker.sh python -m polar.scripts.summarize
```

Without Docker, the same commands run with `.venv/bin/python` (results go to `polar/results/<ds>/<method>_n<labels>_s<seed>/`).
