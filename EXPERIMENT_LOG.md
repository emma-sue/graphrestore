# Experiment Log

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-15 UTC
- Origin Verification Status: VERIFIED THROUGH STAGE2 / PAUSED AT USER APPROVAL GATE
- Origin Version Label: exp_result_v7_1_stage2_pause
- Current Release Status: STAGE4 AND FORMAL MIO100 A/B/C VERIFIED / COMPLETE
- Current Release Label: graphrestore_v7_1_stage4_step040000_formal_complete

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

## FORMAL PREFLIGHT — VERIFIED

- Window: 2026-08-14T23:03:48Z–23:07:38Z.
- Orchestrator result: `PREFLIGHT_COMPLETE`; all nine mandatory commands exited zero.
- Full frozen-tree regression: `123 passed in 47.09s`.
- Metric and degradation parity remained PASS; both real CUDA one-batch cases were finite with 495 gradient tensors.
- Compile A/B ran 20 finite eager optimizer steps. Python 3.12 rejected Dynamo before a compiled optimizer step, so the preregistered fail-closed decision is `torch_compile=false`.
- Independent post-preflight evidence audit recomputed all bound data, manifest, parent, code, config, and GPU-probe hashes with no blocker.

## EXACT-100 INTEGRATION — VERIFIED

- Window: 2026-08-14T23:08:09Z–23:09:35Z.
- Result: exactly 100/100 optimizer steps, finite, `21.3101 img/s`, peak reserved `8,438,939,648` bytes (`33.4055%`).
- Frozen runtime: crop192, effective batch8, micro4, accumulation2, BF16, eager, no gradient checkpointing.
- Worst-phase probe rejected micro8 because its measured peak was `91.2426%`; micro4 was the highest-throughput candidate within the strict 90% ceiling (`16.1369 img/s`, `46.8956%`).
- Recovery artifact: `artifacts/integration/stage0_100_steps/last.pth` plus summary/report/train log.

## STAGE0 — RUNNING

- Formal start: 2026-08-14T23:09:53Z in detached tmux session `graphrestore` through `scripts/orchestrate.py --run_main_pipeline`.
- The formal step-0 raw/resumable anchor was written before training progressed.
- Initial live evidence through step 200: all logged loss and gradient norms finite, approximately `21 img/s`, crop-training peak `33.31%`; no competing GPU process.
- Mandatory terminal boundary remains Stage2: release GPU and pause. Stage3 is prohibited without explicit user approval.

### Step-4000 validation fail-closed event

- Training reached step 4000 with all logged loss, gradients, LR, throughput and state tensors finite. The pre-validation checkpoint was durably written with `pending_validation_step=4000`.
- Full deterministic `primary_val` completed all 1600 images, but the peak CUDA reserved fraction with the complete training state resident was `0.9264`, above the locked `0.9000` ceiling.
- The trainer raised `Stage0ContractError` after metric computation and before publishing any metric JSON, calibration row, best checkpoint or validation commit. The orchestrator entered `FAILED` and released the GPU.
- Recovery evidence remains exact: `last.pth` is raw/resumable at step 4000, sampler consumed step 4000/cursor 32000, and the pending validation must be replayed before another optimizer step.
- No result was silently accepted and no step was lost. An accuracy-neutral allocator/metric-memory A/B is required before resuming; an unchanged blind retry is prohibited.

### Step-4000 allocator recovery A/B

- Full-state checkpoint SHA256: `259af613637e0bfd78a88935aa8de3b96c610c96a4db21048d2a0124cbecc79f`; it remains raw/resumable at step 4000 with pending validation.
- Formal-like first-100 comparison: native default peak reserved `17,540,579,328` bytes (`69.434%`); native expandable segments `9,758,048,256` bytes (`38.627%`). Peak allocated memory was essentially unchanged, while inactive-split peak fell from `4,284,498,432` bytes to zero.
- Controlled deterministic causal comparison: default `24,448,598,016` bytes (`96.780%`) versus expandable `10,766,778,368` bytes (`42.620%`); 100/100 prediction hashes, PSNR and SSIM were exactly equal.
- Two native-default `cudnn.benchmark=true` process runs differed on 9/100 hashes with PSNR max difference `0.00339890 dB`; default versus expandable differed on 17/100 with smaller PSNR max difference `0.00218964 dB`. This establishes that the tiny uncontrolled cross-process variation is normal benchmark algorithm reselection, not allocator-specific arithmetic.
- Frozen recovery environment: `PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`. The 90% ceiling and all numerical/data/checkpoint semantics remain unchanged. Raw evidence is under `artifacts/audits/stage0_validation_allocator/`.

### Step-4000 pending validation replay — VERIFIED

- The resumed process loaded the exact pending raw checkpoint and logged `replay_pending_validation` before any step 4001 update.
- Full replay completed 1600/1600 images, 16 tasks x 100, with internal peak reserved `11,773,411,328` bytes (`46.6050%`), safely below the unchanged 90% ceiling.
- First valid same-protocol incumbent: single PSNR/SSIM `26.1464793694 / 0.8260035340`; Group-A PSNR/SSIM `20.4809440994 / 0.6601006776`. No old ProVIR or formal MiO100 number is used as a baseline.
- Metric artifact SHA256: `57761823...c224`; best-EMA SHA256: `6cc1a621...f65a`; raw-last SHA256 immediately after commit: `4e2124da...2cd4`.
- `best_ema.pth` is `ema_selection/resumable=false` and all 495 `model` tensors are bit-exact to `ema.shadow`; `last.pth` is raw/resumable with pending cleared and sampler step 4000. Calibration history has the exact 28-column schema and one step-4000 row.
- Training automatically continued at step 4001 under the bound allocator environment; post-commit loss, gradients and memory remained finite.

### Step-8000 validation — VERIFIED / NEW BEST

- Training steps 4001–8000 remained finite. Each 1000-step audit had strictly increasing steps, stable throughput, and a flat crop-training peak near `44.87%`; SSIM weight remained exactly zero as required before step 12000.
- The atomic pre-validation checkpoint SHA256 was `140fab8780b659722110f38cf444887145242cc814ebd3f724f15a5f0ef1a6bd`: raw/resumable, step and sampler-consumed step 8000, sample cursor 64000, `pending_validation_step=8000`; model, EMA and optimizer tensors were all finite.
- Full validation completed 1600/1600 images and 16 tasks x 100. Peak reserved was `11,349,786,624` bytes (`44.9281%`), below the unchanged 90% ceiling and without shape-sequence accumulation.
- Step-8000 single PSNR/SSIM: `26.4606376386 / 0.8382246568`; Group-A PSNR/SSIM: `21.3927391911 / 0.6879688255`.
- Relative to the first same-protocol step-4000 incumbent, Group-A improved `+0.9117950916 dB / +0.0278681479 SSIM`, while single improved `+0.3141582692 dB / +0.0122211228 SSIM`.
- The Group-A PSNR gain is well beyond the locked `0.02 dB` tie band, so step 8000 correctly became the new best. `best_ema.pth` is `ema_selection/resumable=false`; all 495 saved model tensors are bit-exact to its EMA shadow. Raw `last.pth` is step 8000 with pending cleared and sampler aligned.
- Metric artifact SHA256: `752047788e2f53da992cbf5ce6c83960a34e8972339574f98b0a2d56c480be35`. Calibration history retained the exact 28-column schema with unique rows for steps 4000 and 8000. Training automatically continued beyond step 8000.

### Step-12000 validation — VERIFIED / NEW BEST

- The pair-heavy curriculum switched at absolute optimizer step 10000 exactly as frozen: deterministic sampler replay gave single/pair `59.9875% / 40.0125%` for steps 9000–9999 and `29.7750% / 70.2250%` for steps 10000–10999, with all 16 tasks represented.
- Training through the step-12000 checkpoint remained finite. The raw/resumable checkpoint committed at step and sampler-consumed step 12000, cursor 96000, with all model/EMA/optimizer tensors finite and pending validation cleared after commit.
- Full validation completed 1600/1600 and 16 tasks x 100 at peak reserved fraction `45.0111%`.
- Single PSNR/SSIM: `26.7962580824 / 0.8457306308`; Group-A PSNR/SSIM: `22.1708072197 / 0.7087221494`.
- Relative to step 8000, Group-A gained `+0.7780680286 dB / +0.0207533239 SSIM` and single gained `+0.3356204438 dB / +0.0075059740 SSIM`; step 12000 correctly became the new best.
- Safe raw recovery checkpoint SHA256: `4ef4c817e925cedc413f2a529c1588f6643e22d07c02696f6585fd92a4b304f8`. An immutable hard-link was preserved as `recovery_step12000_before_ssim_fix.pth` before stopping.

### Post-step-12000 training SSIM numerical failure — FAIL-CLOSED

- Once `lambda_ssim=0.05` became active, 17 of the 25 periodic observations through step 12500 reported a negative `1-SSIM` term; the minimum was `-0.555595`, with a coincident pre-clip gradient norm of `12.7984`.
- Root cause was independently reproduced: `train_ssim_y()` converted its tensors to FP32 but executed its convolutions inside the outer BF16 autocast context. BF16 cancellation in `E[x^2]-E[x]^2` produced impossible SSIM values above one.
- The main process was intentionally interrupted before any post-12000 optimizer state was checkpointed. Orchestration is durably `FAILED/exit 130`, tmux is closed, and the GPU is released. The authoritative resumable state remains the clean step-12000 checkpoint; the 500 transient bad updates will not be resumed.
- Recovery requires an internal-autocast-disabled FP32 training-SSIM implementation, dedicated regression/parity tests, re-running every code-bound gate affected by the source change, and an explicit allowlisted provenance migration whose only state mutation is the provenance binding. Provenance verification will not be disabled.

### Step-12000 FP32-SSIM recovery — VERIFIED / READY TO RESUME

- The corrected training SSIM runs its Y conversion and all moments in an internal FP32/no-autocast region and uses an algebraically equivalent centered/difference formulation. Real CUDA BF16 calls for the core loss and Stage0/1/4 matched their FP32 references exactly; losses and gradients were finite and the SSIM term was nonnegative.
- Code commits: FP32 correction `1e41f8c`; fail-closed migration tool `27bd400`.
- Fresh gates: `137 passed`; Ruff/syntax PASS; official metric parity PSNR `0`, SSIM `3.87908e-7`; validation VRAM Stage0/expanded `36.72% / 37.90%`; compile remains safely disabled; both real CUDA one-batches finite.
- Fresh recovery exact-100 completed 100/100 at `20.2297 img/s`, peak `31.7701%`. Its worst-phase SSIM probe still selected micro4/accum2 (`15.588 img/s`, `44.9945%`) over the slower eligible micro8 (`13.765 img/s`, `86.6518%`).
- Provenance migration permitted exactly three SHA leaves and proved every other checkpoint section bit-exact. Receipts: best SHA `96a70407...3d4a`, raw-last SHA `e42db6c7...5f58`. The migrated best/raw checkpoint SHA values are `4e7d3b0d...71ab3` / `ed2da98d...5ef48`.
- Original best/raw checkpoints remain immutable at `recovery_best_ema_step12000_before_ssim_fix.pth` / `recovery_step12000_before_ssim_fix.pth`; project-native strict loading of both migrated candidates passed before atomic publication. The raw checkpoint remains step12000, sampler cursor96000, pending null, so resume must recompute optimizer input step12000 before any later state.
- Native orchestration resume started at step 12000 under the unchanged expandable-segments allocator. The first completed update was logged explicitly at step 12001 with `lambda_ssim=0.05`, `ssim_loss=0.31628233`, finite gradient, and the exact identity `loss = Charbonnier + 0.05*ssim_loss`.
- Through recomputed step 12500, all 26 new-segment observations had finite, nonnegative SSIM loss and peak reserved `44.83%`. The 25 overlapping step labels 12020–12500 differed in UTC, input-dependent Charbonnier, SSIM, total loss, gradient, and throughput from the discarded run, proving those updates were recomputed rather than resumed.

### Step-16000 validation — VERIFIED / NEW BEST

- The corrected segment from step 12001 through 16000 contained exactly 201 expected log observations. Every SSIM term was finite and nonnegative (minimum `0.0519243`), `lambda_ssim=0.05`, and `loss = Charbonnier + 0.05*SSIM_loss` held with maximum error `3.36e-9`. Crop-training peak was `44.83%`.
- Full validation completed 1600/1600 and 16 tasks x 100. Peak reserved was `11,769,217,024` bytes (`46.5884%`); aggregate and per-task values were all finite and independently recomputed exactly from task means.
- Single PSNR/SSIM: `27.0459342969 / 0.8507952453`; Group-A PSNR/SSIM: `22.9001644909 / 0.7281011114`.
- Relative to step 12000, Group-A improved `+0.7293572712 dB / +0.0193789620 SSIM` and single improved `+0.2496762145 dB / +0.0050646145 SSIM`. The Group-A PSNR margin exceeds the 0.02 dB primary key, so step 16000 correctly became the new best.
- Metric SHA256: `8fdec9206d1bc4e977afb9c66179efcff1a4845b92a917de8ee097a4734dd65b`; best/raw SHA256: `17aada1d...cc9e3` / `7254878d...0948`. Both use current provenance; best model equals EMA shadow for all 495 tensors, raw pending is cleared, and calibration has exactly four 28-column rows at 4000/8000/12000/16000.
- All eight Group-A task PSNRs increased. The equal-weight single aggregate increased despite three individual regressions (motion blur `-0.383981 dB`, noise `-0.959648 dB`, rain `-0.161962 dB`); these are monitored but do not override the locked restoration-first checkpoint rule.

### Step-20000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 16001 through 20000 remained finite and contract-consistent: all 200 expected periodic observations had nonnegative SSIM loss (minimum `0.0721160`), `lambda_ssim=0.05`, and the locked loss identity held with maximum error `2.61e-9`. Crop-training peak remained flat at `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 20000 with `pending_validation_step=20000`, sampler-consumed step 20000/cursor 160000, scheduler/EMA step 20000, current provenance SHA `aa38a917...59200`, and all model/EMA/optimizer/RNG tensors finite.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), safely below the unchanged 90% ceiling; aggregate and per-task metrics were finite and independently reproduced from the task means.
- Single PSNR/SSIM: `27.2506444192 / 0.8549414988`; Group-A PSNR/SSIM: `23.4662212682 / 0.7439277156`.
- Relative to step 16000, Group-A improved `+0.5660567772 dB / +0.0158266042 SSIM` and single improved `+0.2047101223 dB / +0.0041462535 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 20000 correctly became the new best.
- Metric SHA256: `693e7e70d72ec1c9b2b44ea5e37ff6487861e85a0b77135d8d3e019f29b66450`; best/raw SHA256: `b5b41764d5e9280e0b3cc9ad3f8337419d8c5f3bc59e7d4c1b2099ae576712e1` / `5c2b69964faa0af31c3970ed87de73e8be303de657a15b10a1d373da5815aa12`.
- `best_ema.pth` is step-20000 EMA selection/non-resumable; its 495 model tensors are bit-exact to both its EMA shadow and the raw-last EMA. `last.pth` is raw/resumable with pending cleared and sampler cursor 160000. Calibration retains the exact 28-column schema with unique rows at 4000/8000/12000/16000/20000.
- All eight Group-A task PSNRs again increased. The single aggregate increased while motion blur/noise/rain remained below their step-16000 task values; this remains monitored and does not override the preregistered checkpoint-selection rule. Training automatically continued beyond step 20000.

### Step-24000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 20001 through 24000 contained exactly 200 expected periodic observations. Every logged value was finite, SSIM loss stayed nonnegative (`0.0671482`–`0.3701114`), `lambda_ssim=0.05`, and the locked loss identity held within `3.73e-9`; crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 24000 with `pending_validation_step=24000`, sampler-consumed step 24000/cursor 192000, scheduler/EMA step 24000, and current provenance SHA `aa38a917...59200`. Validation left model, EMA, optimizer, scheduler, RNG, sampler and provenance content bit-exact.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), identical to step 20000 and safely below the unchanged 90% ceiling.
- Single PSNR/SSIM: `27.4254196262 / 0.8586839712`; Group-A PSNR/SSIM: `23.8896652555 / 0.7558784358`.
- Relative to step 20000, Group-A improved `+0.4234439874 dB / +0.0119507202 SSIM` and single improved `+0.1747752070 dB / +0.0037424724 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 24000 correctly became the new best.
- Metric SHA256: `e1a0c2e0c7d362a985987fb6ddebbebf89ef6ab5ac4d177dcd0f69a981e343ea`; best/raw SHA256: `75396b68fe1e8a833e77c1d9ab66028064f8645f7fab567bda31291aa87becb7` / `2b41c692481a16453b1151084c6e4edb93dc23049e22304eac9e529feb7fb78d`.
- `best_ema.pth` is step-24000 EMA selection/non-resumable; its model equals both its EMA shadow and the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared and sampler cursor 192000. Calibration has six unique 28-column rows at steps 4000 through 24000. Training automatically continued beyond step 24000.

### Step-28000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 24001 through 28000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss stayed nonnegative (`0.0770500`–`0.3830798`), `lambda_ssim=0.05`, and the loss identity held within `3.92e-9`; crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 28000 with `pending_validation_step=28000`, sampler cursor 224000, scheduler/EMA step 28000, and current provenance SHA `aa38a917...59200`. Validation left model, EMA, optimizer, scheduler, RNG, sampler and provenance content bit-exact.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from steps 20000 and 24000.
- Single PSNR/SSIM: `27.5876459730 / 0.8626043739`; Group-A PSNR/SSIM: `24.2106245899 / 0.7651042136`.
- Relative to step 24000, Group-A improved `+0.3209593344 dB / +0.0092257777 SSIM` and single improved `+0.1622263467 dB / +0.0039204027 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 28000 correctly became the new best.
- Metric SHA256: `4bfbeaa07d75404b6cc5ce093e2747ec797fd2214e095073b234d598d96d04c9`; best/raw SHA256: `05a47d897996e3bdc7fb592b554aaff09ad9c0ba4d264b1ce2eaaee18a885696` / `d72eb6c372849809c963ad2315092bb6cb1cf47c91993c9bf63a642e4568988a`.
- `best_ema.pth` is step-28000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has seven unique 28-column rows at steps 4000 through 28000. Training automatically continued beyond step 28000.

### Step-32000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 28001 through 32000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0672329`–`0.3658830`), `lambda_ssim=0.05`, the loss identity held within `2.61e-9`, and all six locked cosine learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 32000 with `pending_validation_step=32000`, sampler cursor 256000, scheduler/EMA step 32000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from the preceding three validations.
- Single PSNR/SSIM: `27.7342155004 / 0.8665127721`; Group-A PSNR/SSIM: `24.4431806767 / 0.7720839923`.
- Relative to step 28000, Group-A improved `+0.2325560868 dB / +0.0069797787 SSIM` and single improved `+0.1465695274 dB / +0.0039083982 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 32000 correctly became the new best.
- Metric SHA256: `7ba66e1bf38577ebbb159117e426170c6de9b51c347e5169eb346d12a7c4b081`; best/raw SHA256: `a457f7d62a66f8203817215d32c1ba470c6cea9a7cc212ef57781d3a06d23d04` / `77ef00e104ea1db4f4722a3f82a67b1c85a55101bf10e78b07d266d67a5db078`.
- `best_ema.pth` is step-32000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has eight unique 28-column rows at steps 4000 through 32000. Training automatically continued beyond step 32000.

### Step-36000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 32001 through 36000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0785243`–`0.4040532`), `lambda_ssim=0.05`, the loss identity held within `3.73e-9`, and all six locked learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 36000 with `pending_validation_step=36000`, sampler cursor 288000, scheduler/EMA step 36000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from every validation since step 20000.
- Single PSNR/SSIM: `27.8437229085 / 0.8694816782`; Group-A PSNR/SSIM: `24.5939226604 / 0.7768620701`.
- Relative to step 32000, Group-A improved `+0.1507419837 dB / +0.0047780778 SSIM` and single improved `+0.1095074081 dB / +0.0029689060 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 36000 correctly became the new best.
- Metric SHA256: `587aaa388328de418551c95a52590c10963053e8d01e062280f79f3ee30e67d5`; best/raw SHA256: `57f0dbc01ec295e97a028e609747415c06e94be2181b20d339610725721abe15` / `1d9d1bb8f7754ea19c77f9e13acae17c5d101a341294b99a27f64fcb9de17282`.
- `best_ema.pth` is step-36000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has nine unique 28-column rows at steps 4000 through 36000. Training automatically continued beyond step 36000.

### Step-40000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 36001 through 40000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0477487`–`0.3823910`), `lambda_ssim=0.05`, the loss identity held within `3.73e-9`, and all six locked learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 40000 with `pending_validation_step=40000`, sampler cursor 320000, scheduler/EMA step 40000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from every validation since step 20000.
- Single PSNR/SSIM: `27.9586054242 / 0.8718410850`; Group-A PSNR/SSIM: `24.6742421770 / 0.7800438344`.
- Relative to step 36000, Group-A improved `+0.0803195167 dB / +0.0031817643 SSIM` and single improved `+0.1148825157 dB / +0.0023594068 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 40000 correctly became the new best.
- Metric SHA256: `e68596e314a0edefe90cfbc6c8a64bba2aa6404b97138f0a57714604b5c9a4f4`; best/raw SHA256: `b385ebf27bf23247a8c0127b125f46037836f8d5f4b74ab61b63cfd3618d40bf` / `b213111a9a04a9730c09424ec28dce945527a8a6c5d04f5f0b4deeca63bbd92a`.
- `best_ema.pth` is step-40000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has ten unique 28-column rows at steps 4000 through 40000. Training automatically continued beyond step 40000.

### Step-44000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 40001 through 44000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0610270`–`0.3861921`), `lambda_ssim=0.05`, the loss identity held within `3.17e-9`, and all six locked learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 44000 with `pending_validation_step=44000`, sampler cursor 352000, scheduler/EMA step 44000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from every validation since step 20000.
- Single PSNR/SSIM: `28.0418478096 / 0.8734113923`; Group-A PSNR/SSIM: `24.7245536053 / 0.7821832080`.
- Relative to step 40000, Group-A improved `+0.0503114283 dB / +0.0021393735 SSIM` and single improved `+0.0832423854 dB / +0.0015703073 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 44000 correctly became the new best.
- Metric SHA256: `757fff260837338e81c3a1a0f751037370fef21bf788f2ef7cc0e2aa5ea1fa6d`; best/raw SHA256: `c0b6d909e380948ad3c3c35106fd97d4de93a34d3a467d7237c04cb9b2e49077` / `a08d51126bab94e202bcafeab8346801988518af1b9817edf39a3055379e47c7`.
- `best_ema.pth` is step-44000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has eleven unique 28-column rows at steps 4000 through 44000. Training automatically continued beyond step 44000.

### Step-48000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 44001 through 48000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0541608`–`0.4407394`), `lambda_ssim=0.05`, the loss identity held within `2.99e-9`, and all six locked learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 48000 with `pending_validation_step=48000`, sampler cursor 384000, scheduler/EMA step 48000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from every validation since step 20000.
- Single PSNR/SSIM: `28.1208996296 / 0.8746030128`; Group-A PSNR/SSIM: `24.7605605412 / 0.7837054803`.
- Relative to step 44000, Group-A improved `+0.0360069358 dB / +0.0015222723 SSIM` and single improved `+0.0790518200 dB / +0.0011916206 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 48000 correctly became the new best.
- Metric SHA256: `de7d6bfb1fa5aee1f9cd630ea626d1a19847f92391b2a6851dd910bea30adfd8`; best/raw SHA256: `ee546963ac28476cdf2e09c5f91da0a73271171e248544a9754ecc00c905e27b` / `f3b4f288c21d0808c67a6cf2f810d4f0c23c49bda9a6eb79abec35e0bc16952a`.
- `best_ema.pth` is step-48000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has twelve unique 28-column rows at steps 4000 through 48000. Training automatically continued beyond step 48000.

### Step-52000 validation — VERIFIED / NEW BEST

- The corrected FP32-SSIM trajectory from step 48001 through 52000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0489299`–`0.3288177`), `lambda_ssim=0.05`, the loss identity held within `7.83e-9`, and all six locked learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 52000 with `pending_validation_step=52000`, sampler cursor 416000, scheduler/EMA step 52000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from every validation since step 20000.
- Single PSNR/SSIM: `28.1908498490 / 0.8755784694`; Group-A PSNR/SSIM: `24.7905910957 / 0.7847861019`.
- Relative to step 48000, Group-A improved `+0.0300305545 dB / +0.0010806216 SSIM` and single improved `+0.0699502194 dB / +0.0009754566 SSIM`. The Group-A PSNR margin exceeds the locked `0.02 dB` primary key, so step 52000 correctly became the new best.
- Metric SHA256: `3dd9f42fb9b3928f3a612f74296b6e7ea0db279ebe471b955e1695dfd189a634`; best/raw SHA256: `7c878bb2ced548f43f4f8eeffdd91e26d65ba680e5ae3c58d591affe34b39998` / `45e06bef0ebf26a267e6fdb00a90e9b71e10020a468a6b7cdb1c53febdb681c0`.
- `best_ema.pth` is step-52000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has thirteen unique 28-column rows at steps 4000 through 52000. Training automatically continued beyond step 52000.

### Step-56000 validation — VERIFIED / NEW BEST VIA SSIM TIE-BAND

- The corrected FP32-SSIM trajectory from step 52001 through 56000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0674261`–`0.3552809`), `lambda_ssim=0.05`, the loss identity held within `6.34e-9`, and all six locked learning rates recomputed exactly. Crop-training peak remained `44.6625%`.
- The raw/resumable pre-validation checkpoint was committed at step 56000 with `pending_validation_step=56000`, sampler cursor 448000, scheduler/EMA step 56000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Full validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`), unchanged from every validation since step 20000.
- Single PSNR/SSIM: `28.2246385992 / 0.8762187060`; Group-A PSNR/SSIM: `24.8042141533 / 0.7854323259`.
- Relative to step 52000, Group-A improved `+0.0136230576 dB / +0.0006462240 SSIM` and single improved `+0.0337887502 dB / +0.0006402366 SSIM`. Because the Group-A PSNR difference is inside the locked `<0.02 dB` tie band, the higher Group-A SSIM correctly made step 56000 the new best.
- Metric SHA256: `748f3db89ca6a89c547726ecc49e23e9418f656cd2fd423045542a230ad543dc`; best/raw SHA256: `f2f0955fd8745abec995b92ee759436a4965bd5f310557855c47acb121aa8561` / `aa46f64762aea77dc86987aaca1e566dc823da6cfa029bbb21011bc656e63e17`.
- `best_ema.pth` is step-56000 EMA selection/non-resumable and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has fourteen unique 28-column rows at steps 4000 through 56000. Training automatically continued beyond step 56000.

### Step-60000 final validation and Stage0 completion — VERIFIED / FINAL BEST

- The final corrected FP32-SSIM trajectory from step 56001 through 60000 contained exactly 200 expected periodic observations. All fields were finite, SSIM loss remained nonnegative (`0.0422670`–`0.3302498`), `lambda_ssim=0.05`, the loss identity held within `3.73e-9`, and all six learning rates reached the locked `1e-6` endpoint exactly. Crop-training peak remained `44.6625%`.
- The final raw/resumable pre-validation checkpoint was committed at step 60000 with `pending_validation_step=60000`, sampler consumed/cursor `60000 / 480000`, scheduler and EMA step 60000, and current provenance SHA `aa38a917...59200`. Pre/post deterministic hashes for model, EMA, optimizer, scheduler, RNG, sampler and provenance were all identical.
- Final validation completed 1600/1600 images and exactly 16 tasks x 100. Peak reserved was `11,286,872,064` bytes (`44.6791%`); all aggregate and per-task metrics were finite and independently reproduced from the task means.
- Final single PSNR/SSIM: `28.2524459159 / 0.8767248622`; final Group-A PSNR/SSIM: `24.8097213721 / 0.7859094886`.
- Relative to step 56000, Group-A improved `+0.0055072188 dB / +0.0004771627 SSIM` and single improved `+0.0278073168 dB / +0.0005061562 SSIM`. Because the Group-A PSNR difference is inside the locked `<0.02 dB` tie band, the higher Group-A SSIM correctly made step 60000 the final best.
- Metric SHA256: `c67ecdcd444090da2163f96176a6e8dc40d39681907fd99d928a4eaa982f40fc`; best/raw SHA256: `52a8744582e39e4f1aa052cc84924ad486289c0b97fc30c89fc6489e69dfac8a` / `6b2acd82484029b83e2eff047a3d78b89e2a61bd256571007d81a6d78d7b779a`.
- Final summary SHA256: `a83ba62ec0a79a09f3a0715e044e25e1a41c5e5690c8f2df7313852825ca6a90`; report SHA256: `0fceb2de785f97db76a88b00ac9bca01d445c8729889dfc3c68175ed85f3e837`. The summary records `completed_step=target_step=60000`, finite=true, overall train/validation peaks below 90%, and matches the terminal complete event.
- `best_ema.pth` is the step-60000 EMA selection/non-resumable checkpoint and equals the raw-last EMA for all 495 tensors. `last.pth` is raw/resumable with pending cleared. Calibration has fifteen unique 28-column rows at steps 4000 through 60000. Stage0 exited zero.

### Stage1 automatic start — VERIFIED

- Orchestration started Stage1 only after the Stage0 final commit and zero exit. The sole GPU process inherited the locked expandable-segments allocator.
- The Stage1 step-zero anchor binds the exact final Stage0 best SHA. All 495 shared backbone tensors are bit-exact to the Stage0 best EMA; the 1040 new tensors are exclusively under the allowed `decoder.skill_bank.*` prefix, with no unexpected or shape-mismatched keys.
- The Stage1 step-zero raw checkpoint is resumable with sampler cursor zero, scheduler/EMA updates zero, and model/EMA tensors finite and bit-exact. Runtime probing selected micro-batch 8, accumulation 1, effective batch 8: the highest-throughput eligible candidate, with peak `67.9235%` below the locked 90% ceiling.
- Stage1 run-contract SHA256: `23338eba488f8ee07537955238af39d96b0035f2cbb26422b6863df0ec73c8f7`; step-zero checkpoint SHA256: `ad7951e4df4103a8db7c929317d5fdcffc013b5f61428ae8ecaaa6e7abd80c21`.

### Stage1 step-3000 EMA-scope audit — REJECTED BEFORE VALIDATION COMMIT

- Raw training/optimizer semantics were correct through step 3000: the 495
  shared backbone tensors remained bit-exact to the final Stage0 EMA, and Adam
  state existed only for the 1040 permitted skill/mixer parameters. The
  checkpoint was raw/resumable with `pending_validation_step=3000`.
- The generic EMA nevertheless rounded unchanged frozen FP32 state on every
  update. At the pending boundary, 479/495 shared EMA tensors differed from raw
  (`5,450,213 / 25,437,220` elements, max absolute `1.860857e-4`); 82,308
  elements remained different after BF16 projection. Because validation uses
  EMA weights, the uncommitted result could not be attributed to an exactly
  frozen Stage0 backbone.
- Stage1 was intentionally interrupted before publishing any validation
  metric/best/report/calibration row. The interrupt was outside an optimizer
  update; the checkpoint remained complete and readable. The orchestrator
  exited 130, tmux ended, and the GPU was released.
- The entire rejected canonical Stage1 directory was atomically archived at
  `artifacts/archives/stage1_rejected_ema_scope_20260815T155215Z/stage1`.
  Archive receipt SHA256:
  `623f562123aa3e89bcb14cb7ce648ac20e0ce80e2d4a98185bd5598ae1f5f944`.
  The archived last/contract/train-log SHA256 values are respectively
  `6ce99c8a...7be0d`, `23338eba...c8f7`, and `924e734a...1116`.

### Stage1 phase-aware EMA remediation — AUDITED / HARD-STOPPED BEFORE RESTART

- Implemented a Stage1-only dynamic-scope EMA: standard FP32 EMA for currently
  trainable named parameters; exact copy for frozen parameters and all buffers.
  The phase-0 end-exclusive boundary is explicitly 5000, and the first
  post-unfreeze update uses the ordinary EMA formula without a shadow reset.
- Formal training and micro-batch probing share the same EMA factory. New
  checkpoint scope/policy/count/dtype gates are enforced before any state
  mutation; Stage2/Stage3 reject old or forged Stage1 EMA parents.
- Added an exact sparse optimizer-state name ledger. It prevents silent Adam
  history loss while preserving legitimate teacher-forced sparsity; valid
  step-1 sparse, step-0 empty, and step-5001 newly-unfrozen states pass.
- Independent fail-closed counterexamples cover cleared/deleted Adam state,
  bad ledger IDs/names/roles, phase-local step overflow, static/dynamic
  optimizer and scheduler drift, malformed moments, missing CUDA RNG,
  sampler/metrics/provenance/EMA mismatch, and downstream stage forgery. All
  are rejected before model/EMA/optimizer/scheduler/sampler/RNG mutation.
- Verification: affected `65 passed`; full CPU suite `183 passed`; Ruff,
  `git diff --check`, and compileall pass. Code commit:
  `75ec9d44c518fd1a9326989665468c358e1122e8`.
- Stage0 anchors were rehashed unchanged. The canonical Stage1 checkpoint
  directory is absent; orchestration remains `FAILED` only because of the
  deliberate exit 130; no trainer/orchestrator/tmux/GPU process exists.
- Per user instruction, the audited clean-restart command has **not** been
  executed. Work is hard-stopped immediately before a fresh Stage1 step-0
  launch pending external GPT review and explicit user approval.

## CANONICAL STAGE1 FRESH RUN — VERIFIED / COMPLETE

- The approved fresh retry started at `2026-08-15T17:35:42Z` from an empty canonical Stage1 directory. The earlier rejected EMA-scope trajectory and the one-ULP pre-anchor LR-validator initialization are archived and contribute no canonical metric or checkpoint.
- Stage1 completed all 30,000 optimizer steps and ten locked 3,200-evaluation validations. Every validation was selected as an improvement under the frozen Group-A PSNR/SSIM/single ordering rule.
- Final step-30000 Group-A PSNR/SSIM: `25.470635026693344 / 0.772928579242125`; single: `28.985509393215178 / 0.8669213048307631`; pair-isolation: `29.241925800442697 / 0.8468425488412182`.
- From the first canonical step-3000 validation to final: Group-A `+2.241846318245 dB / +0.034807630652`; single `+4.376440523863 dB / +0.056069471404`; pair-isolation `+4.940822417736 dB / +0.074471515571`.
- Training and validation maximum reserved fractions were `0.7531189278515263` and `0.4691220305653632`, both below the locked 0.90 ceiling. All logged losses, gradients, FP32 SSIM terms and learning rates passed the continuous numerical audit.
- Final best/raw/validation SHA256: `433bcab29f21c98f42107ad6d1c3f8214848254a7ef4d6ca7d6a2141da5bfcaa` / `e7b5a090ea377b6723ac903ef38c74bf24d80fa1572aa5f7011fec3d95154bf1` / `981a51b3f6a88bdf50f349f53c3fd97a8a248774e0ad1a3a6856c0248b0b1f5b`.
- Final state audit: `best.model == last.EMA == best.EMA` for 1535/1535 tensors; optimizer/state/name-ledger exact at 1423 names; 112 permanently frozen tensors/buffers remain bit-exact to the Stage0 parent; scheduler reaches step/count `30000/30001`, all three LRs reach `1e-6`, and sampler cursor `240000` exactly exhausts the epoch.
- Stage1 `complete.json` SHA256 is `569545f041e17c85b9e307ce60a5a11278bed0f1a17e2ea4172ca9e14c88d7b2`; completion provenance SHA256 is `c852975870a6cac75996875f37e419835febb29d705c2615c8ace174445f2a1e`. Stage2 and Stage3 production loaders independently accepted the same final parent without initializing CUDA.

## STAGE2 EFFECT PROFILES AND INTERACTION DISTILLATION — VERIFIED / COMPLETE

- Effect-profile inference completed `4096 = 512 source samples × 8 forced skills` records in 256 atomic shards. Raw/aggregate SHA256: `fe6d69bc6e782240301c9c0c1a8d9608035d20caf7f8de4b0785ee1c19f0ce2d` / `a1af32c5c285591a0eb18cd21c89a82a43f11b2fac19aefb0d6c10d1c3d0ef07`. The result is 40D, inference-only, optimizer-free, and records zero MiO100 or Group-B/C exposure.
- Interaction manifests contain 4096 train and 800 val samples. Clean IDs are 2496/276 unique with zero train/val overlap. All 4,896 relation records were recomputed with the production label rule: finite, unique, same-start, same canonicalization and exact Stage1 parent binding.
- Train labels: i-before-j 778, j-before-i 815, parallel 2010, ambiguous 493. Val labels: 113, 146, 476, 65. Ambiguous samples retain `class_index=None, weight=0.25` and are excluded from pair priors and all non-ambiguous descriptive statistics.
- Val decision statistics: ambiguous fraction `0.08125`; non-ambiguous parallel fraction `0.6476190476190476`; serial-gap PSNR median/P75 `0.19983482360839844 / 1.1096587181091309 dB`; median pair-majority share `0.607929292929293`. Seven of eight pair-majority labels agree across train/val; `motion_blur+low_light` is the sole context-dependent mismatch and is reported without automatic threshold/model changes.
- Relation train/val SHA256: `b65de122964c963636c94b14fac7494c9c995c3f9c4022e405d4d6d026642c18` / `6c641406fc50e26a5e1af30b4d113a00439576341b5620022e5ab8514c189f30`. Pair-prior/global-priority SHA256: `4116725bce4ecfaceaa1429183e86738ee7ec38835e25886cacd9aa3aec38d82` / `80504122f5ce8e8beedf630426bd8e15485efc9db0f9b3d0adf39ca6dd54b0d8`.
- Stage2 decision/summary/report SHA256: `434e209ac0db201ca7f1be045e3811547d8f3cc974ff3ef740c96c3689329a47` / `06ec62420273acbd85ddb65751279f884a97725830d2803923a66cfc4ac143ce` / `6631942335487f05f27a4bde22c58918565507fd11bb9f51ea6c43163daf7a5c`. Only informational context/stability flags are present; `approved=false`, `stage3_started=false`, and no automatic model change occurred.
- `STAGE3_APPROVAL_REQUIRED.json` SHA256 is `33be4aba2c4229175ac33edef7a5914a48a249b8c733d86338c64a8662072825`; all 22 frozen bindings rehash exactly. `STAGE3_APPROVED.json` is absent, and the production Stage3 preflight rejects before CUDA initialization.
- Stage2 exited zero and persisted `PAUSED_AFTER_STAGE2`; GPU compute processes, orchestrator/trainer processes and tmux are all absent. The required five-line `RUNNING_STATUS.md` pause block is exact; its SHA256 is `5c5f1c101d244ae26fd36f1bd65cc2fdcfdbb860c9d166586787cc8a621ee9e3`.
- Terminal state for this authorized segment: GPU released, Stage3 NOT STARTED, waiting for explicit user approval. The only legal continuation command remains `python scripts/orchestrate.py --approve_stage3 --resume_from_stage3`; it has not been executed.

## STAGE3 PRELAUNCH AUTHORIZATION AND HARDENING — PASS

- User authorization received on 2026-08-17 UTC: D-017 ordinal `severity/2` accepted; only Stage3→Stage4 authorized; formal MiO100 remains prohibited.
- CPU-only final verification: Stage3/Stage4/orchestration `31/21/58` passed (110 targeted); full repository `259 passed`; Ruff, format check for all 9 modified Python files, compileall and diff-check passed.
- Frozen evidence rehashed: Stage1 best `433bcab2…fcaa`, Stage2 decision `434e209a…9a47`, approval-required `33be4aba…2825`; 22/22 physical bindings exact.
- Prelaunch provenance receipt: `reports/STAGE3_PRELAUNCH_PROVENANCE_RECEIPT.json`, SHA256 `6c565c81cd61c3aabd96c6035f6a18859265ea0e571c6d807ea818de0f4f05e0`.
- Material Passport command (single authorized instance): `/root/miniconda3/bin/python scripts/orchestrate.py --approve_stage3 --resume_from_stage3`, cwd `/root/autodl-tmp/aaa/graphrestore`, child allocator `backend:native,expandable_segments:True`.
- Expected automatic outputs: atomic `STAGE3_APPROVED.json`; Stage3 raw/best/thresholds/report/complete; Stage4 raw/best/report/diagnostics/complete; terminal GPU release and pause before formal MiO100.
- Failure policy: no silent retry; any non-zero child exit or evidence mismatch is reported and remains fail-closed.
- Started UTC: `2026-08-17T09:29:13Z`; tmux session `graphrestore`; orchestrator PID `519375`; initial Stage3 child PID `519412`.
- Atomic approval SHA256: `7b351c0958aa681dc1f65114e801c58e3a5bc4bb7cc73c06507c0b647e51a08b`. Its D-017 adjudication, Stage3→Stage4-only scope, formal-MiO100=false boundary, approval-required SHA and Stage2 decision SHA were read back exact.
- Parent and child `/proc` environments both contain exactly `PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`; orchestration state is `STAGE3_RUNNING`, `last_error=null`.

## STAGE3 STEP-2000 VALIDATION FAIL-CLOSE — GUARD DIAGNOSTIC PADDING

- Training steps 1–2000 were continuous and finite with exact scheduler LR; peak reserved fraction was `0.04599072817788201`.
- The durable step-2000 raw checkpoint SHA256 is `526b4a2d43695204c33fa99f88278a69512eb7d7213583680b21d0258a054921`; it remains resumable with `pending_validation_step=2000`. EMA, the 103-entry planner Adam ledger, sampler cursor 16000, RNG, parent tensors and provenance all passed CPU-only audit.
- After all 1600 validation forwards, the non-ranking guard diagnostic rejected a shape mismatch and Stage3 exited 3 at `2026-08-17T11:30:10Z`. No best, validation, report, threshold or partial artifact was published; the GPU and tmux were released.
- Deterministic cause: full-size inputs are padded on the right/bottom to a multiple of 8 before planner inference, while GT guards are pooled at original-image H/4. Example: 1356×2040 yields predicted 340×510 versus target 339×510. The intended remediation only crops predicted diagnostic padding with a strict per-axis delta of 0 or 1; it does not alter model execution, training, restoration metrics or selection.

## STAGE3 PENDING-2000 CONTROLLED PROVENANCE MIGRATION — VERIFIED

- The Stage3/Stage4 diagnostic alignment fix was validated by 283 repository CPU tests, 62 Stage3/Stage4 tests, 14 migration tests, Ruff, compileall and diff-check. It only removes right/bottom padding from non-ranking guard diagnostics; model execution, training loss, restoration metrics and selection remain unchanged.
- At `2026-08-17T12:11:09Z`, the raw pending checkpoint and run contract were atomically migrated. Exactly two semantic-source leaves changed: `stage3_engine.py` and `stage4_engine.py`; all other 19 checkpoint sections and non-provenance contract content are bit-exact to the read-only backups.
- New run-contract/checkpoint SHA256: `156a57b5f74659c45d2123e98c3e89c02b4611136e960d1134d0d88b092084b5` / `39bc85036a372df040774bf93d3000d0a5e36853e0e07b4648d7a01953a30d16`. Migration receipt SHA256: `449bd49b3e31a430eed1d4c6e217c4299084beb272d9845648ded95b7f8718e6`.
- Independent CPU-only review confirmed step/pending 2000, EMA/model/optimizer/scheduler/sampler/RNG exactness, 47/47 current semantic sources, 22/22 approval bindings, unchanged approval artifacts, no partial outputs and no CUDA initialization. The checkpoint is cleared for the canonical post-approval resume path.

## STAGE3 PENDING-2000 RESUME DEVICE CHECK — FAIL-CLOSED BEFORE MUTATION

- The canonical post-approval resume was attempted once at `2026-08-17T12:17:20Z`. It exited 1 after 12 seconds inside the pre-mutation EMA/frozen-parent validator because a CPU-mapped checkpoint tensor and a CUDA live-model tensor were passed directly to `torch.equal`.
- No model, EMA, optimizer, scheduler, sampler or RNG state was installed. The migrated run-contract/checkpoint SHA256 values remain `156a57b5...84b5` / `39bc8503...0d16`; their inode and mtime are unchanged. The training log has no resume row, and no best, validation, threshold, report, temporary or partial artifact exists.
- Orchestration is fail-closed with exit 1 and GPU released. The same command must not be retried until exact cross-device parent comparison is fixed, regression-tested and reflected through a new controlled semantic-source provenance migration.

## STAGE3 PENDING-2000 EMA-DEVICE PROVENANCE MIGRATION — VERIFIED

- The frozen-parent comparison now moves a live CUDA reference tensor to the CPU checkpoint device only when necessary, preserving exact shape/dtype/layout/value equality without casting, tolerance or mutation. A real CUDA resume succeeds; a synchronized raw/EMA frozen-tensor forgery is rejected before model, EMA, optimizer, scheduler, sampler or RNG mutation.
- Verification passed 312 CPU tests with one expected CUDA-only skip, the dedicated real-CUDA test, 28 migration fault tests, Ruff, compileall and diff-check.
- At `2026-08-17T12:43:59Z`, a second independent atomic migration changed only the `src/training/stage3_engine.py` semantic-source leaf. New run-contract/checkpoint SHA256: `d98b7493b41a0ace9fcb228c50b3acbdf855f092bb2ddc9c9f479730cecf053f` / `07489be40a3ed43153024cf8a4f9450c3c5f2f96e2bf0e44e8db9f3ef5c2fc63`; receipt SHA256: `9848708c1a2dc91a99230a68ebf630c8574c64b6cbc8bad97700b5846efc21cb`.
- Stage4 and the other 46 semantic sources, all non-provenance checkpoint state, approval and 22 bindings, orchestration failure state and the first COMPLETE migration receipt are unchanged. Read-only same-filesystem backups preserve the pre-migration anchors.

## STAGE3 PENDING-2000 VALIDATION REPLAY — RUNNING

- Independent production-loader review accepted the migrated checkpoint. The canonical post-approval resume was launched once at `2026-08-17T12:54:28Z` with the locked native expandable allocator.
- At `12:54:38Z`, the child durably logged `resume(step=2000,pending_validation_step=2000)` followed immediately by `validation_replay`. No step 2001 or new training prefetch preceded the replay, so the completed 2,000 optimizer updates were not rerun.
- The orchestrator and sole Stage3 GPU child are active. The pending raw checkpoint remains the validation transaction anchor until the full 1,600-image result commits atomically.
- At the user's request, active Codex monitoring was stopped to save agent compute while the independent tmux/orchestrator/trainer chain continues unchanged. The last handoff check found `STAGE3_RUNNING`, `last_error=null`, one healthy GPU child, and the log still exactly at `validation_replay(step=2000)` with no step 2001 or early artifact publication.

## STAGE3 STEP-12000 EXTENSION AUTHORIZATION — MIGRATED / PRE-LAUNCH

- Stage3 completed all original 12,000 optimizer steps and six validations. The selected step-12000 autonomous result is single `25.405096 / 0.830990`, Group-A `22.027225 / 0.711882`, and macro-F1 `0.845490`; all 12,000 train rows are continuous and finite.
- Final presence calibration then failed before thresholds/`complete.json` because its direct encoder path omitted the model's formal pad-to-8 operation on non-multiple-of-8 full-resolution inputs. The fix applies the same formal padding only to this presence-collection path; it does not change training, model weights, restoration metrics, or checkpoint selection.
- The user explicitly authorized three additional Stage3 train/validation cycles from the latest raw/resumable step-12000 state: validations at steps `14000`, `16000`, and `18000`. The original cosine horizon remains `12000`; the already-reached learning-rate floor remains exactly `2e-6` for the extra 6,000 steps.
- Extension hardening passed the final CPU repository suite (`398 passed`, one expected CUDA-only skip), the 35-case migration fault suite, Ruff, format, compileall, and `git diff --check`. No Stage0/1/2 state, original approval, scientific config, data, or 22 frozen bindings changed.
- At `2026-08-18T04:35:38Z`, the controlled three-file migration completed atomically. It created the separate flat-20 extension authorization SHA256 `43e010f9c66301415b8bd2d3ac7e48aa7653283671a2756dc744926a4a4724fd`, added the exact target/schedule provenance, and changed exactly four frozen semantic-source leaves.
- New live run-contract/raw-last/EMA-best SHA256 values are `0f38784922de0670a2974f234de22b1c464495fd3c268d08128e29205ed26311`, `e15d544ac1e9459ce97b0ed3f0992eded61824f1cb144d44174b425d1d23b6b1`, and `9114974f68f202119d4241077d0c46333315204959d58b7eabecaf68a3e32ff3`. Migration receipt SHA256 is `a5fe047b065542e825ba39a1729d94ffe76eba0aa268d5cfaa81a3213867a9a1`.
- All 19 non-provenance sections of both checkpoints are type/metadata/raw-byte exact to three same-disk `0444` backups. Raw model, 103-state Adam, EMA, scheduler (`last_epoch=max_steps=12000`, LR `2e-6`), RNG, sampler (`12000/96000`), metrics and selected best remain unchanged. Post-migration Stage3 and orchestrator authorization loaders pass without CUDA initialization.
- At `2026-08-18T04:45:55Z`, the exact post-approval extension command started in tmux `graphrestore` with `PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`. Orchestrator PID `8924` launched the sole Stage3 GPU parent PID `8994` with canonical `--resume last.pth --extension_authorization STAGE3_EXTENSION_APPROVED.json` arguments.
- The trainer logged `resume(step=12000)` and then step `12001`; it did not replay prior training or the already-committed step-12000 validation. Initial extension rows are finite, use exactly LR `2e-6`, and remain below the 0.90 VRAM ceiling.

## STAGE3 FINALIZATION REVOCATION AND STAGE4 LAUNCH — VERIFIED / RUNNING

- The user superseded the temporary Stage3 extension authorization and froze the step-12000 selected EMA as the only Stage4 parent. The unselected step-14000 raw/pending state was archived as audit evidence and never entered validation history, calibration, selection, or Stage4 loading.
- The immutable revocation SHA256 is `a4d3abb112aae01afeef5c87f3ffb65a26f3ff8cdff1cc8ad362db2323d9743e`. It permanently rejects plain or extension Stage3 training before CUDA/optimizer creation while authorizing finalize-only calibration and Stage4.
- Dedicated Stage3 finalization executed zero optimizer steps, checkpoint writes, or sampler advances. Threshold SHA256 is `1030a9708cb802e8ec993de12fb926ea4e48360b35251ddc642548dfd2aa260e`; all eight skills satisfy calibrated F1 >= F1@0.50, and macro-F1 improved `0.8454899764 -> 0.9086336704`.
- The independent calibrated 1600-image diagnostic SHA256 is `52dd68fa62f07840f3b59e16253ed3945b2a3453ec083137b3fc12afe253486c`: single `25.5378938150 / 0.8390662570`, Group-A `22.6091022968 / 0.7331561477`, relation accuracy `0.6231292517`, with zero MiO100/Group-B/Group-C reads.
- Stage3 complete/report SHA256 values are `2c666c2a02baf8e5b188b869083a215fd392802f31f61d2022293b52e54b9d17` and `c6a260ca524c5b254ebcf59c507b4785f936350ce7c71f03601e8321b5ec5116`. All frozen best/selected/history/run/train anchors remain unchanged.
- Stage4 started automatically at `2026-08-18T08:43:48Z`. Run-contract SHA256 is `46aca21b891b5da7194546a04a44d156d713315c06965c53b53e6334e14ca0ab`; it binds the step-12000 Stage3 best, frozen thresholds, Stage1 lineage, finalization outputs, primary-only data boundary, 40,000-step schedule, and formal-MiO100=false.
- Launch audit passed: step0 raw/resumable anchor, fresh optimizer over 1,526 trainable parameters, exact frozen digest, Stage3/Stage1 tensor lineage, 10-step probe, and both 2040-square topology gates (peak `49.693%`). Early training peak is `9.0155%`.
- Stage4 steps 1–2000 are continuous and finite. Four 500-step loss medians are `0.08872, 0.08231, 0.08110, 0.07903`; Planner medians are `0.64507, 0.58910, 0.56252, 0.54988`. LR/SSIM schedules and deterministic teacher flags are exact, all eight skills remain active, and no forbidden-data descriptor or Stage4 error is present.
- Yellow watch items for the first step-4000 validation: isolated pre-clip gradient spikes are contained by clip=0.5; low-resolution guard strength is increasing without global saturation; motion-blur precision, low-light activation, unexpected activation, and guard high-fraction require full-validation review. None currently meets a stop condition.

## STAGE4 STEP-4000 CALIBRATION-LEDGER RECOVERY — VERIFIED / TRAINING RESUMED

- The first 1,600-image step-4000 validation finished inference, but the shared
  calibration writer mistook legitimate Stage0 and Stage3 rows at the same step
  for duplicate Stage4 rows and failed closed. The raw checkpoint remained
  exactly at `step=4000`, `pending_validation_step=4000`; the log contained no
  validation event or step 4001, so no optimizer step was lost or repeated.
- Deviation D-020 introduces the Stage4-owned sidecar ledger without changing
  any model computation, data, metric, selection rule, or schedule. The two
  controlled provenance migrations have COMPLETE receipt SHA256 values
  `795982a5f607c147e25a2553a63c4b24306fd0fe2753cdcd6ca0cab0af8c190d`
  and `3be807f84ec1b4d12141bfb4de75040b0124706eefe863eeabfebfa2c140bdcd`.
  The combined migration/Stage4 regression suite passed 110 tests; all
  non-provenance checkpoint state remained bit-exact.
- Pending validation replay committed atomically at
  `2026-08-18T16:07:51Z`. The sidecar SHA256 is
  `c81df5da7bc0399455a47045e5b61a084b871b36548d36667783399a6c1a5620`
  and contains exactly its header plus one step-4000 row. The frozen shared
  history remains SHA256
  `b282987c3f77034f76788a412e91823cd4570ce8c6c10cd93030ee181612e034`.
  Committed `validation_latest.json`, raw-last and EMA-best SHA256 values are
  `d8d1b78e1ca329d12c00fa488eb2ac5953c2d732f65591a6841b3f94c0d3dd09`,
  `b931d9cac96e6df7ac4bf0623ead2c0ff9ddf6cf91ca4dd96cb5b9a179dd7071`,
  and `833f2fb72c22b2d2cd9b166a647171bd84ef2aa316c1809f41f2babf7231bbff`.
  Raw-last has `pending_validation_step=null`; the train log has exactly one
  replay event followed by one validation event before step 4001.
- The authoritative replay result is Single
  `25.7236196601 / 0.8427595733` and Group-A
  `22.7981546116 / 0.7449925725`. Relative to the frozen Stage3 calibrated
  parent this is `+0.185726 dB / +0.003693 SSIM` and
  `+0.189052 dB / +0.011836 SSIM`, respectively. Planner macro-F1 is
  `0.9085288995` and relation accuracy is `0.6612244898`. This is an early
  positive restoration trend, not the final `+0.30 dB` success threshold and
  not yet a claim against the Stage0 scientific baseline.
- Stage4 resumed at step 4001 immediately after the durable commit and remains
  on the locked 40,000-step schedule. Formal MiO100/Group-B/Group-C reads remain
  zero.

## FORMAL MIO100 GROUP A/B/C — VERIFIED / COMPLETE

- Stage4 completed at step 40000 and froze the EMA selection checkpoint SHA256
  `6aa2de6e65ce633430d188857845acef1b67cfb3b218a04977293dbc149a84fd`.
  The conditional extension gate observed only `+0.000370104312900 dB`
  Group-A PSNR from step 36000 to 40000, so it durably selected
  `DO_NOT_EXTEND`; no step 40001 was executed.
- The explicitly authorized, one-shot formal evaluation used the frozen
  online-canonical MiO100 manifest and completed all 1,440 images: Group A
  `8 x 80 = 640`, Group B `4 x 100 = 400`, and Group C `4 x 100 = 400`.
  Autonomous graph inference was retained; task-label routing, TTA, model soup,
  threshold tuning, and result-driven reruns were disabled.
- Official CUDA six-metric group means are: A
  `24.3741960555 / 0.7599264849 / 0.2731640404 / 0.2741912930 / 0.4126837323 / 53.9287128568`;
  B `19.8229607308 / 0.6678735027 / 0.3619331664 / 0.2726837824 / 0.3968436596 / 50.5320779753`;
  C `18.5247367167 / 0.5563379351 / 0.5058757574 / 0.1970546311 / 0.2750860956 / 38.0631561780`,
  in PSNR/SSIM/LPIPS/MANIQA/CLIP-IQA/MUSIQ order.
- All 144 immutable scoring shards were complete, contiguous and finite; the
  maximum scorer reserved fraction was `0.150009469`. The original controller
  failed closed only because an auxiliary CPU/CPU small-sample observed maximum
  had been misapplied as a CPU/CUDA tolerance. Prediction and GT identities
  were exact for all 1,440 rows. A separately approved CPU-only finalizer
  published the existing official-CUDA shard results without reading images,
  recomputing metrics, changing a tolerance, or initializing CUDA.
- Strict terminal verification returned `COMPLETE` for 1,440 images and 144
  shards. Per-image/summary/complete SHA256 values are
  `f8bcbd463eb7113ccf632b5e03f0a34951650a8b0e66e1e63edabbfdd781a6d5`,
  `68301ea9f0ef52f062a8fdaa763865c3fb0374ee49f65ec9d99cdd9cf5ac7be8`,
  and `eb1468705d2591709565b3461aa630ae826f1efc311cddaf6bc06feca309a60b`.
  Recovery receipt SHA256 is
  `fc3aec42a22f06fe866adf988aa2a25ddb591bf7ef54ef1868f9c550cecb7ce9`.
  No evaluator/scorer/recovery process remains and the GPU is released.
- Scientific conclusion: Group A has strong PSNR/SSIM and exceeds the
  published AgenticIR and RAR values on those two metrics, but Group B is below
  both methods and Group C is mixed/weak. This does not support a blanket SOTA
  or broad compositional-generalization claim. Full report:
  `reports/MIO100_FINAL.md` (SHA256
  `129e582fddfeb34249c8af2ae6302cb13b4e9e8d0a461fbb27ef8cb86cda6682`).
