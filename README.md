# GraphRestore

Implementation workspace for **Plan the Repair, Restrain the Edit: Partial-Order Guarded Skill Programs for Composite Image Restoration**.

The normative implementation contract is:

`/root/autodl-tmp/graphed/GUARDED_GRAPHRESTORE_FINAL_V7_1_AGENTICIR_CODEX_PROMPT.md`

This workspace uses only the frozen MiOIR clean/depth training manifests and AgenticIR Group-A/single recipes. Group B/C and MiO100 formal images are excluded from training, calibration, checkpoint selection, and early stopping.

Main pipeline:

```text
Stage0 -> Stage1 -> Stage2 -> mandatory pause/release GPU
user approval -> Stage3 -> Stage4
user authorization after complete freeze -> MiO100 A/B/C formal evaluation
```

See `RUNNING_STATUS.md`, `reports/DEVIATIONS.md`, and `EXPERIMENT_LOG.md` for the live state and audit trail.

