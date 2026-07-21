# Migration plan from the legacy workspace

The original workspace remains outside this Git repository and is treated as a
read-only evidence source.

## Legacy files used as behavioral references

| Legacy path | Purpose | SHA-256 prefix |
|---|---|---|
| `data_input.py` | CWRU loading/filtering/windowing | `EACE606B962F` |
| `feature_pipeline_sliding_tstks.py` | 151-d feature contract and old KS features | `7CEFD60BE595` |
| `model_training_final_v2.py` | single-run teacher/Diamond baseline | `0F2C37826DBF` |
| `tri_train.py` | historical three-run training variant | `56697EDBAA29` |

These scripts are not copied as maintained code because they mix absolute paths,
data leakage, training, evaluation, and output generation. Required behavior is
migrated into tested modules in small stages.

## Stages

1. TSTKS reference implementation and synthetic tests.
2. Source-level CWRU/SEU manifest generation and leakage tests.
3. Reproducible legacy teacher/student baselines.
4. Learnable gate and confidence-weighted distillation.
5. Compact Top-K export and actual conditional feature computation.
6. Experiments, aggregate tables, figures, and manuscript result registry.

Old CSV figures and model checkpoints are classified as `legacy_unverified` and
will not be used as final manuscript evidence.

