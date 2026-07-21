# Data directory

Raw datasets are not committed to this repository.

The rebuild will create source-level manifests under `data/manifests/`. Every
window must retain the following fields:

- `sample_id`
- `source_id`
- `condition_id`
- `window_start`
- `split`
- `label`
- `sampling_rate`
- `raw_path`

Train/validation/test assignment is performed on a source file or continuous
time segment before any overlapping window is produced.

