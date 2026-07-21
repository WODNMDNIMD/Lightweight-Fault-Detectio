# Sliding-window TSTKS implementation note

## Source basis

The implementation is a clean-room reconstruction based on:

1. Zou Junchen, Qi Jinpeng, Li Na, Liu Jialun, Zhu Houjie. “Design and
   Implementation of a Fast Online Algorithm for Mutation Point Detection.”
   *Electronic Science and Technology*, 2020, 33(8): 10–15.
   DOI: <https://doi.org/10.16180/j.cnki.issn1007-7820.2020.08.002>
2. The later open-access RSW&TST description, which gives explicit
   trigeminal-branch definitions and search criteria:
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC8694460/>

No source code accompanied the supplied six-page paper. This repository does
not claim byte-for-byte reproduction of the authors' unpublished program.

## What is directly supported by the paper

- Long time-series data are processed window by window to support multiple
  change points.
- The TST search tree has left, middle, and right branches.
- The middle branch overlaps both side branches, reducing boundary misses.
- The final decision uses a two-sample Kolmogorov–Smirnov statistic scaled by
  `sqrt(m*n/(m+n))` and compared with a threshold.
- Window width trades detection quality against runtime and must be reported,
  not silently tuned on a test set.

## Explicit engineering choices in this repository

- A 2048-point bearing sample is scanned with 256-point analysis windows and a
  128-point hop by default. This retains the earlier workspace's local-analysis
  scale while replacing its exhaustive flat search with a trigeminal path.
- Adjacent child branches overlap by 50% of a child width.
- Branch navigation can use scaled KS, normalized variance fluctuation, or a
  hybrid of both. `hybrid` is the default; final acceptance always uses KS.
- `1.36` is the conventional asymptotic two-sample KS critical value for an
  approximate 5% level. It is a configurable starting value, not a claimed
  dataset-optimal threshold. A final threshold must be calibrated on training
  data and selected with validation data only.
- Optional peak alignment is disabled by default because it is not specified
  in the supplied paper. It remains configurable for a documented ablation.

## Complexity

Let `W` be the analysis-window width, `H` the hop, `d` the tree depth, and `s`
the candidate split step. A flat exhaustive implementation evaluates roughly
`W/s` candidates per window. The trigeminal implementation evaluates candidates
inside three shrinking branches at each level. Its current reference
implementation prioritizes clarity and determinism; profiling and an optimized
ECDF update are separate work items.

## Terminology boundary

Within the fault-diagnosis paper, these outputs are described as **local
distribution-change statistical features**. They are not presented as verified
ground-truth fault onset times unless an external onset annotation exists.

