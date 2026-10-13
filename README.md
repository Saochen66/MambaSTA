# MambaSTA V8

Source code and configuration for the MambaSTA V8 `J1_R4` multimodal EEG-fNIRS model.

## Architecture

`J1_R4` keeps the frozen B_M50 EEG and fNIRS feature branches and replaces probability averaging with a single reference-centred joint classifier. The deployed head combines:

- an EEG main effect;
- an fNIRS main effect;
- a rank-4 EEG-fNIRS interaction term.

B_M50 probabilities are used only as a training teacher and do not enter the deployed joint logits.

## Repository contents

- `models/`: V8 model and inherited backbone dependencies.
- `training/`: cached-feature, staged, baseline and shared training utilities.
- `tools/`: V8 orchestration, frozen evaluation and preprocessing entry points.
- `configs/`: the frozen experiment configuration and a portable path template.

No datasets, checkpoints, prediction files, experiment outputs or logs are included.

## Configuration

Copy `configs/mambasta_v8_portable.example.yaml` and update the dataset, experiment and Python paths for your system. The frozen configuration is retained separately for protocol transparency.

The expected evaluation protocol uses three held-out sessions and seeds `42`, `123` and `2026`. Model selection is validation-only; test outputs must not be generated before the selection freeze.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install `mamba-ssm` and `causal-conv1d` only when compatible accelerated kernels are available for the local PyTorch/CUDA environment.

## Entry points

```bash
python tools/v8_run.py --help
python tools/v8_s1.py --help
python tools/v8_optimize.py --help
python tools/v8_finalize.py --help
```

The MI/MA validation-selected Top10 extension is available through:

```bash
python tools/v8_mi_ma_top10_extension.py --help
```

## Data

The dataset is not redistributed in this repository. Prepare locally authorized EEG-fNIRS data and point a copied portable configuration to those paths.

