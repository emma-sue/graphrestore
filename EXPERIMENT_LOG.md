# Experiment Log

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-15 UTC
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## PRE-STAGE0

- Status: implementation and frozen-identity audit in progress.
- Formal training has not started.
- Expected first command: `python scripts/audit_data.py`.
- Monitoring contract: process-alive, log growth, GPU/VRAM, loss/metric finite checks, and explicit hard timeout per run.

### Reference environment audit

- Exact AgenticIR NumPy 2.1.2 + OpenCV 4.9 import: failed due binary ABI mismatch.
- Corrective isolated reference: NumPy 1.26.4 + OpenCV 4.9.0 + BasicSR 1.4.2 + pyiqa 0.1.10.
- Import/imresize/PSNR/SSIM sanity: passed (`PSNR(0,0)=80`, `SSIM(0,0)=1`).
- Official locked-reference parity passed on 16 full-size and 8 native-x4 pairs: PSNR max absolute error 0, SSIM max absolute error `3.87908013377e-07`, and both float and uint8 x4 canonical byte hashes exact on all 8 pairs.
- Evidence: `reports/METRIC_PROTOCOL.md`, `artifacts/metrics/metric_parity_per_image.csv`, and `artifacts/metrics/metric_parity_summary.json`.
- Metric/canonicalization verification status: VERIFIED for the mandatory parity sample; end-to-end training remains PRE-STAGE0.

### Official degradation cross-environment parity

- Command: `python scripts/audit_degradation_parity.py`.
- Data: two frozen primary-train single recipes for each of the eight official operators (16 total); no MiO100 image was read.
- Result: all 16 BGR uint8 outputs were byte-identical between the training environment and the isolated AgenticIR reference environment.
- Evidence: `reports/DEGRADATION_PROTOCOL.md` and `artifacts/audits/degradation_parity.json`.

### Mandatory real-data one-batch CUDA checks

- `python tests/test_one_batch.py --case single`: PASS, rain sample, loss `0.01113312598`, 495 finite gradient tensors, peak reserved fraction `0.12602`.
- `python tests/test_one_batch.py --case group_a_low_resolution`: PASS, motion-blur + low-resolution sample, loss `0.15945617855`, 495 finite gradient tensors, peak reserved fraction `0.12403`.
- Both used crop 192, BF16 forward, the exact 495-key parent, and the online native-to-BasicSR float path where applicable.

### Preregistered Stage0 torch.compile A/B

- Command: `python scripts/profile_stage0_compile.py --config configs/stage0_mio_stagea.yaml`.
- Design: one fixed real primary-train crop-192 batch of eight, steps 0–19, identical seed/parent/optimizer/BF16/TF32; step 0 excluded from steady-state throughput.
- Eager result: all losses/outputs finite, steady-state `24.51787045 img/s`, peak reserved `22,590,521,344` bytes (`89.42%` of the bound RTX 4090 allocation).
- Compiled result: safely rejected before any compiled optimizer step because Torch 2.3 Dynamo does not support the installed Python 3.12 runtime.
- Frozen decision: eager. This is the preregistered safe fallback, not a failed Stage0 gate; no dependency downgrade or environment mutation was performed.
- Evidence: `artifacts/audits/stage0_compile_ab.json` and `reports/STAGE0_COMPILE_AB.md`. The result is bound to config, parent checkpoint, profiler SHA, Torch/CUDA/GPU identity, crop and batch, and is idempotently revalidated.

### Cross-stage checkpoint correction

- A read-only cross-stage audit found that the initial Stage1 `best_ema.pth` path saved raw model weights under `payload["model"]`.
- Corrected before any formal training: best saves now install EMA weights during the atomic serialization, while `last.pth` remains the raw resumable model; regression asserts bit-exact `best.model == best.ema.shadow`.
- Stage1 + Stage2 directed regression after the correction: `10 passed`.

### Full-resolution validation VRAM gate

- Command: `python scripts/probe_validation_vram.py --force`.
- Locked largest `clean_val` geometry: `2040x2040` (`clean_id=000017`, four-way maximum-area tie resolved by manifest order).
- Stage0 MiOStageA BF16 inference plus the exact quantized official GPU PSNR/FP64-SSIM path: finite, peak reserved `11,096,031,232` bytes (`43.9236%`).
- Expanded two-skill GuardedSkillRestormer BF16 inference: finite, peak reserved `13,730,054,144` bytes (`54.3504%`). Stage1 official metrics run on CPU, matching its validator.
- Both are below the locked `0.90` ceiling. Evidence is bound to the maximum image, config, resolved paths, parent checkpoint, hardware and semantic code hashes in `artifacts/audits/validation_vram_probe.json`.

### Exact-resume and persistence hardening

- `last.pth` now carries `model_role=raw_training_state,resumable=true`; `best_ema.pth` carries `model_role=ema_selection,resumable=false`. Resume rejects selection snapshots before changing model, optimizer, scheduler or RNG.
- Stage0/1 write a step-0 recovery anchor. Validation boundaries are transactions with `pending_validation_step`: interruption at an intermediate or final validation replays that exact validation before any new optimizer step and clears pending only after metrics/report/best/last are durable.
- Training and validation CUDA peaks are measured independently. Full-resolution validation can no longer poison the crop-training peak counter.
- Stage1 refuses to serialize a mid-optimizer-update signal state; it falls back to the prior atomic checkpoint. Durable stale `*_RUNNING` orchestration state can be explicitly resumed only after confirming no exact child process remains alive.
- Text/JSON/CSV/shard renames now fsync their parent directory. Stage2 success evidence includes labels, priors, decision, summary CSV and report; approval rechecks every declared Stage2/config SHA.
- Current full CPU regression before formal preflight: `123 passed`; full Ruff and syntax checks pass. Formal preflight will rerun all tests after the final code freeze.
