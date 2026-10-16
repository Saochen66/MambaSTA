# MambaSTA

MambaSTA is a multimodal EEG-fNIRS classifier. The released implementation is
the validation-selected `J1_R4` model: frozen modality encoders feed a compact,
reference-centred joint head with EEG and fNIRS main effects plus a rank-4
cross-modal interaction.

## Model

For EEG feature vector `e`, fNIRS feature vector `n`, and learned reference
vectors `e0` and `n0`, the classifier contains:

- an EEG main-effect term;
- an fNIRS main-effect term;
- a low-rank interaction between `(e - e0)` and `(n - n0)`;
- one shared class-logit output.

The historical B_M50 model supplies frozen encoders and training-time teacher
probabilities. Its probabilities are not averaged into the released inference
logits. `models/mambasta.py` is the stable public model entry point.

## Repository contents

- `models/`: MambaSTA and the inherited encoder dependencies.
- `training/`: cached-feature and staged training utilities.
- `tools/`: preprocessing, fitting and frozen-evaluation entry points.
- `configs/final.yaml`: portable three-seed reproduction configuration with
  seed 42 marked as the sole reported execution.
- `configs/ablation.yaml`: predeclared inference-ablation configuration.

This repository intentionally contains code and configuration only. It does
not contain datasets, subject identifiers, checkpoints, cached features,
predictions, result tables, experiment logs, or private absolute paths.

## Installation

Create an environment compatible with your CUDA/PyTorch installation, then:

```bash
python -m pip install -r requirements.txt
```

The Mamba encoder requires compatible `mamba-ssm` and `causal-conv1d`
packages. The implementation does not silently replace Mamba with another
architecture.

## Configuration and protocol

Copy `configs/final.yaml`, then set local data and output paths outside the
repository. The public configuration exposes seeds `42`, `123`, and `2026`
for independent reproduction. The reported experiment in this release was
actually executed once with seed `42` only; it is not a three-seed robustness
study. It uses three held-out sessions, validation-only model selection, and
a test freeze before test export.

The ablation configuration defines four evaluations from the same frozen
checkpoint: full fusion, interaction removal, EEG-reference intervention, and
fNIRS-reference intervention. The reference interventions are diagnostic
inference ablations, not independently retrained unimodal networks.

The public trainers consume a per-fold JSON contract derived from
`configs/final.yaml`. Set `model_seed` to one of the three published values;
use `42` to reproduce the metrics reported in this release.

Useful entry points include:

```bash
python -m training.train_v8_cached --help
python -m training.train_v8_staged --help
```

## Data and claims

The EEG-fNIRS datasets are not redistributed. Users must obtain authorized
copies and reproduce the preprocessing contract locally. Event-onset timing
was not independently established in this release, so the model outputs must
not be interpreted as causal or true-HRF/physiological-peak evidence.

