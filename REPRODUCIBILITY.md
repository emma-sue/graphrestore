# GraphRestore reproducibility archive

This repository preserves the exact source tree, frozen configurations,
manifests, selected checkpoints, metric protocol, and principal result
artifacts for GraphRestore V7.1. Raw datasets, generated degraded images,
formal prediction PNGs, optimizer recovery snapshots, Python environments, and
metric-model caches are intentionally excluded.

## 1. Clone and retrieve checkpoint objects

```bash
git lfs install
git clone https://github.com/emma-sue/graphrestore.git
cd graphrestore
git lfs pull
sha256sum -c WEIGHTS.sha256
```

The five expected objects are:

| Role | Path | Step | SHA256 |
|---|---|---:|---|
| Stage-A warm-start | `artifacts/checkpoints/stage_a/final_backbone.ckpt` | — | `66e056ff3537ea99416aeb119173e90fbcafc9e9f809db169ef7381cc93f77b8` |
| Stage0 selected EMA | `artifacts/checkpoints/stage0/best_ema.pth` | 60000 | `52a8744582e39e4f1aa052cc84924ad486289c0b97fc30c89fc6489e69dfac8a` |
| Stage1 selected EMA | `artifacts/checkpoints/stage1/best_ema.pth` | 30000 | `433bcab29f21c98f42107ad6d1c3f8214848254a7ef4d6ca7d6a2141da5bfcaa` |
| Stage3 selected EMA | `artifacts/checkpoints/stage3/best_ema.pth` | 12000 | `9114974f68f202119d4241077d0c46333315204959d58b7eabecaf68a3e32ff3` |
| Stage4 selected EMA | `artifacts/checkpoints/stage4/best_ema.pth` | 40000 | `6aa2de6e65ce633430d188857845acef1b67cfb3b218a04977293dbc149a84fd` |

The `best_ema.pth` files are selection snapshots (`resumable=false`), intended
for inference, evaluation, lineage verification, and warm-starting the next
stage. Large raw optimizer/RNG recovery snapshots are not part of this public
archive.

## 2. Exact filesystem layout

The completed run deliberately embeds absolute paths and verifies physical
source/config hashes. For byte-for-byte contract replay, clone to:

```text
/root/autodl-tmp/aaa/graphrestore
```

and place data/upstream repositories below:

```text
/root/autodl-tmp/graph/data/graphrestore
/root/autodl-tmp/graph/training_data
/root/autodl-tmp/graph/upstream/AgenticIR
/root/autodl-tmp/graph/upstream/MiOIR
```

The original Stage-A files were external. To reproduce the frozen layout
without changing `configs/resolved_paths.yaml`, materialize them at:

```text
/root/autodl-tmp/aaa/provir/artifacts/checkpoints/stage_a/final_backbone.ckpt
/root/autodl-tmp/aaa/provir/artifacts/manifests/stage_a_final_selection.json
```

using the archived files under `artifacts/checkpoints/stage_a/`. Changing the
paths or frozen configs is possible for a new run, but produces a new protocol
identity and must not be presented as the original run.

## 3. Data and upstream identities

Datasets are not redistributed. Acquire them from their original sources. The
exact training/validation manifest metadata is archived under
`manifests/training/`; it contains paths, recipes, seeds, and hashes, but no
image/depth bytes. The frozen identities are recorded in
`configs/resolved_paths.yaml`, `manifests/`, and `reports/DATA_AUDIT.md`.

Upstream source revisions:

```text
AgenticIR: 9640a291480dee3ba8f2974125d4ee9e3440f3d6
MiOIR:     4d5f6ca0235cf2c307319673242d5722ee35d73f
```

Training uses only the frozen primary-train/primary-val single and Group-A
recipes. RAR, DIV2K, Flickr2K, MiO100 exploration-160, and MiO100 Group B/C
are not training inputs. Formal B/C are read only by the separately authorized
final evaluator.

Run the data identity audit after installing the datasets:

```bash
/root/miniconda3/bin/python scripts/audit_data.py
```

## 4. Environments

The main run used Python 3.12.3, PyTorch 2.3.0+cu121, torchvision
0.18.0+cu121, CUDA runtime 12.1, NumPy 2.4.6, OpenCV 5.0.0, and an RTX 4090.
Direct Python package versions are in `environment/main-requirements.txt`.
Test and lint tooling is pinned in `environment/dev-requirements.txt`.

The isolated AgenticIR reference scorer uses the pins in
`environment/reference-requirements.txt`; setup details are in
`environment/README.md`. Run the parity gates before training or evaluation:

```bash
/root/miniconda3/bin/python scripts/audit_metric_parity.py
/root/miniconda3/bin/python scripts/audit_degradation_parity.py
```

The repository retains `artifacts/formal_mio100/cache/weights_lock.json` as
the identity ledger for the third-party AgenticIR/pyiqa metric weights, but it
does not redistribute the roughly 1.5 GB metric cache. A strict six-metric
rerun therefore requires restoring every locked cache object at the recorded
path and SHA256. If those exact objects are unavailable, populate a new
data-disk cache and create a new protocol identity rather than reusing the
historical approval or claiming a byte-exact rerun.

## 5. Pipeline entry points

The canonical training sequence is described in the archived V7.1 contract
and `EXPERIMENT_LOG.md`:

```text
Stage0 -> Stage1 -> Stage2 -> mandatory approval pause
       -> Stage3 -> Stage4 -> separately authorized formal MiO100 evaluation
```

Primary entry points:

```text
scripts/train_stage0.py
scripts/train_stage1_skills.py
scripts/build_skill_effect_profiles.py
scripts/distill_interactions.py
scripts/train_stage3_planner.py
scripts/train_stage4_e2e.py
scripts/orchestrate.py
scripts/eval_mio100.py
scripts/score_agenticir_table1.py
```

The approval files in the archive are historical evidence. Do not reuse a
machine-bound authorization after changing any path, source, checkpoint,
manifest, dependency, or metric weight; generate a new protocol identity and
approval instead.

## 6. Verification without datasets

After cloning and pulling LFS objects:

```bash
sha256sum -c WEIGHTS.sha256
/root/miniconda3/bin/python -m compileall -q src scripts
/root/miniconda3/bin/python -m pytest -q
```

The full test suite contains CUDA/data-dependent gates; on a host without the
frozen data or GPU, run the pure configuration/model tests and interpret skips
accordingly. The final scientific values and their evidence hashes are in
`reports/MIO100_FINAL.md`.

## 7. Result boundary

The formal Stage4 result is strong on familiar Group-A combinations but weak
on unseen Group-B/C composition. It is not a blanket SOTA claim. A formal
same-protocol Stage0 A/B/C evaluation and fair A1/A3/A4 retraining controls
remain outside this archived result and must not be inferred from the Stage4
table.
