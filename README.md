# GraphRestore

Implementation and reproducibility archive for **Plan the Repair, Restrain the Edit: Partial-Order Guarded Skill Programs for Composite Image Restoration**.

The normative implementation contract is archived at:

`docs/GUARDED_GRAPHRESTORE_FINAL_V7_1_AGENTICIR_CODEX_PROMPT.md`

This workspace uses only the frozen MiOIR clean/depth training manifests and AgenticIR Group-A/single recipes. Group B/C and MiO100 formal images are excluded from training, calibration, checkpoint selection, and early stopping.

Main pipeline:

```text
Stage0 -> Stage1 -> Stage2 -> mandatory pause/release GPU
user approval -> Stage3 -> Stage4
user authorization after complete freeze -> MiO100 A/B/C formal evaluation
```

## Released checkpoints

Git LFS stores the frozen Stage-A warm-start and the selected EMA checkpoints
for Stage0, Stage1, Stage3, and Stage4. The final model used for the formal
MiO100 evaluation is:

```text
artifacts/checkpoints/stage4/best_ema.pth
step: 40000
sha256: 6aa2de6e65ce633430d188857845acef1b67cfb3b218a04977293dbc149a84fd
```

Run `sha256sum -c WEIGHTS.sha256` after cloning to verify every LFS object.

## Formal MiO100 result

The frozen Stage4 checkpoint was evaluated once on the AgenticIR online-
canonical MiO100 test set (1,440 images; A/B/C = 640/400/400). The official
CUDA six-metric group means are:

| Group | PSNR | SSIM | LPIPS | MANIQA | CLIP-IQA | MUSIQ |
|---|---:|---:|---:|---:|---:|---:|
| A | 24.3742 | 0.7599 | 0.2732 | 0.2742 | 0.4127 | 53.93 |
| B | 19.8230 | 0.6679 | 0.3619 | 0.2727 | 0.3968 | 50.53 |
| C | 18.5247 | 0.5563 | 0.5059 | 0.1971 | 0.2751 | 38.06 |

See `reports/MIO100_FINAL.md` for all 16 combinations, comparisons, caveats,
and immutable evidence hashes.

## Reproducing the project

The datasets and generated prediction images are deliberately excluded. See
`REPRODUCIBILITY.md` for the exact repository layout, upstream commits,
environment, data identities, checkpoint roles, verification commands, and
training/evaluation entry points.

See `RUNNING_STATUS.md`, `reports/DEVIATIONS.md`, and `EXPERIMENT_LOG.md` for
the final state and complete audit trail.
