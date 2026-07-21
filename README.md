# C-TGSD: Lightweight Fault Diagnosis

Reproducible research code for cost-constrained teacher-guided sparse
distillation (C-TGSD) in vibration-based fault diagnosis.

This repository is being rebuilt from an earlier experimental workspace. The
rebuild follows three rules:

1. split continuous sources before windowing;
2. keep every reported value traceable to a saved run artifact;
3. preserve legacy behavior as a baseline without treating legacy results as
   verified evidence.

## Current status

- Repository bootstrap completed.
- A paper-aligned, clean-room sliding-window TSTKS implementation is included.
- Synthetic unit tests cover branch overlap, single/multiple change points,
  constant signals, determinism, and the 20-dimensional feature contract.
- Dataset manifests, leakage-safe preprocessing, model training, and C-TGSD
  experiments are the next staged changes.

## Layout

```text
configs/            experiment configuration
data/manifests/     generated source-level split manifests (data excluded)
docs/               method notes, migration decisions, and references
src/ctgsd/          maintained Python package
tests/              dependency-light unit tests
outputs/            generated runs/tables/figures/logs (contents excluded)
```

## Run the current tests

The first-stage tests only require NumPy:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Data policy

Raw CWRU/SEU data, generated NPY arrays, model checkpoints, and experiment
outputs are intentionally excluded from Git. Dataset acquisition and manifest
generation will be documented instead of uploading large or redistributed
third-party data files.

