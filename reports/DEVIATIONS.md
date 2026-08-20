# V7.1 Deviations and Deterministic Interpretations

No scientific model, data boundary, optimization budget, or metric definition is changed here. This file resolves implementation-level omissions or contradictions in the V7.1 contract so that one deterministic code path exists.

## D-001: online canonical manifest semantics — `SPEC_CLARIFICATION`

- Contract conflict: Section 2.5 requires `mio100_test_1440_agenticir_online_canonical.jsonl`; Section 17.2 shows `mio100_test_1440_agenticir_canonical.jsonl`.
- Resolution: the builder and all formal evaluation code use the explicit Section 2.5 `*_agenticir_online_canonical.jsonl` names. These are derived metadata manifests, not pre-generated PNG manifests. Every row retains `native_lq_path`, `gt_path` or target-size metadata, `contains_low_resolution`, `native_scale`, and `input_mode=agenticir_online_canonical`; `canonical_lq_path` is absent or null. The CLI rejects the shorter stale name rather than silently resolving it.
- Runtime rule: low-resolution rows decode official native BGR uint8, convert to float, apply locked BasicSR x4, clamp, convert to RGB float, and feed that float directly to Restormer without another uint8 rounding. Other rows decode native LQ directly to RGB float. No canonical PNG is materialized and no legacy OpenCV canonical path is a formal model input.
- Scientific impact: none; this enforces native-LQ online BasicSR canonicalization and prevents use of existing OpenCV canonical files.

## D-002: ambiguous relation partial-label semantics — `SPEC_CLARIFICATION`

- Contract conflict: Stage2 emits `label=ambiguous, relation_weight=0.25`, while Stage3 relation logits have only three legal classes.
- Resolution: the relation head remains exactly three classes: `i_before_j`, `j_before_i`, and `parallel`. Non-ambiguous rows use ordinary one-hot CE with weight 1. Ambiguous rows use the stable partial-label serial-mass contribution `-0.25 * logsumexp(log_softmax(logits)[0:2])`. Batch normalization is `sum(nonambiguous CE + 0.25 * [-log(serial mass)]) / (n_nonambiguous + 0.25*n_ambiguous)`; the factor 0.25 is applied exactly once.
- Audit exclusion: ambiguous rows do not enter non-ambiguous relation accuracy, parallel precision/recall, pair-prior direction/parallel frequencies, Stage2 majority-label share, or `parallel_fraction_nonambiguous`. They are reported separately as `n_ambiguous` and `ambiguous_fraction` and may still provide presence/guard/stop supervision.
- Prohibited mappings: no fourth logit, mapping to parallel, random serial direction, or program-score argmax pseudo-label.
- Scientific impact: supervises only the supported assertion that serial probability mass should be high, without fabricating an ordering.

## D-003: compile once plus dynamic execution feedback — `SCIENTIFIC_DESIGN_LOCK`

- Contract conflict: Section 9.3 defines one initial DAG at `t=0`; Section 11.1 pseudocode calls the compiler inside the round loop.
- Resolution: at `t=0`, run the Planner over all eight skills, select at most the top three active skills, predict relations for active pairs, compile exactly one acyclic DAG, and retain immutable initial nodes, edges, and topological structure plus mutable `executed`/`skipped` state. At `t>0`, re-encode the current image and update only presence, local guards, and stop. Never recompute relation logits, add nodes, alter edges, or recompile the full DAG.
- Execution: take the earliest unfinished topological level of the fixed DAG; mark nodes below their frozen current-presence threshold `skipped_without_execution`, execute the remainder, and remove both categories from pending state. Updating zero-indegree availability after removal is graph-state advancement, not recompilation. No re-entry is permitted and each skill can be called at most once. `Kmax_train=2`; `Kmax_test=3`.
- Scientific impact: this is the locked GraphRestore main-version design of one relation plan with dynamic execution feedback. It is not protocol reproduction and not an engineering fallback.

## D-004: stop decision

- Contract omission: no stop-logit inference threshold or precedence is specified.
- Resolution: at each round, STOP when `sigmoid(stop_logit) >= 0.5` and no unexecuted skill exceeds its frozen per-skill threshold. Otherwise use the specified presence rule: thresholded top-3; if none pass and max probability `<0.15`, STOP; if max is `>=0.15`, force top-1.
- Scientific impact: the stop head cannot suppress a confidently present remaining skill; clean/no-op states can terminate.

## D-005: forced counterfactual execution

- Contract conflict: counterfactual episodes force an absent skill but normal execution multiplies its guard by a presence probability supervised toward zero.
- Resolution: for the explicitly forced counterfactual skill only, override the execution presence gate to one while retaining the Planner's absent presence target; use its predicted spatial guard without GT. This permits no-op losses to train guard/adapter/executor restraint under a real forced call.
- Scientific impact: implements the stated purpose of joint counterfactual calibration without using true task labels at inference.

## D-006: low-light guard target

- Contract omission: low-light is conditional in the dense list but absent from the global fallback list.
- Resolution: use the contract-provided dense formula `clamp(1 - Y_lq/(Y_clean+eps), 0, 1)` computed from the actual official before/after operator images, with the same crop/augmentation.
- Scientific impact: no new heuristic; this selects the only formula provided by the contract.

## D-007: Stage3 snapshot selection before one-time threshold calibration

- Contract tension: all-stage checkpoint ordering is restoration-first, but per-skill threshold calibration is allowed only once after Stage3.
- Resolution: Stage3 validation uses a fixed, preregistered 0.5 presence threshold for restoration trajectory/checkpoint ordering. After choosing the Stage3 snapshot, perform the specified one-time per-skill calibration on `primary_val` and freeze it.
- Scientific impact: avoids repeated validation tuning and preserves restoration-first selection.

## D-008: missing configs/tests and fair Parallel-Only retraining

- Add the required `configs/baselines/global_guard.yaml` and include `tests/test_checkpoint_resume.py` in the mandatory test command.
- `parallel_only.yaml` is implemented at scaffold time. If Full is effective, a compute-matched Parallel-Only retrain is scheduled alongside the other fairness ablations because the contract states zero-training diagnostics do not replace fair retraining.
- Scientific impact: closes missing deliverables; does not alter the Full model.

## D-009: remaining deterministic defaults

- `L_step=0` for episodes with no intermediate state.
- Cycle loss averages all valid triples among the eight skills, masked to relevant active/present skills.
- Stage2 sampling uses sorted stable recipe IDs with seed 2027 and writes explicit train/val manifests plus SHA256.
- Fallback global-priority edges use zero confidence with stable skill-index tie-breaking; prior-directed edges use their class margin.
- Stage4 teacher probability retains the written discontinuity from 0.5 before step 12000 to 0.25 at/after step 12000.
- Training-time Restormer padding/crop-back follows the warm-start implementation and is tested explicitly.
- Hash mismatches are hard stops; stages are never silently regenerated or rerun.
- The tmux shell enables `pipefail`; the orchestrator streams once to its own
  append-only log, avoiding a second `tee` writer to the same file.

## Pending empirical definitions

The following will be frozen before the affected run and appended here: `torch.compile` numerical tolerance, exact Stage2 sample manifest SHA values, and the preregistered numerical rule for “Full effective” before expensive fair retraining. None may use MiO100 B/C.

## D-010: executable AgenticIR reference environment

- The locked AgenticIR requirements combine `opencv_python==4.9.0.80` (a wheel compiled against NumPy 1.x) with `numpy==2.1.2`; this fails at import with `_ARRAY_API not found` / `numpy.core.multiarray failed to import` on Python 3.12.
- The isolated `.venv-reference` therefore uses NumPy 1.26.4 with the contract-pinned SciPy 1.14.1, OpenCV 4.9.0, BasicSR 1.4.2 and pyiqa 0.1.10. It inherits Torch 2.3.0+cu121 from the untouched training environment.
- BasicSR 1.4.2 imports the removed `torchvision.transforms.functional_tensor`; the reference entrypoint installs an in-memory compatibility alias to `torchvision.transforms.functional.rgb_to_grayscale`. This does not alter degradation mathematics.
- This environment is only authoritative after the mandatory metric, degradation, and low-resolution pixel-parity audits pass. All versions and this deviation are recorded in the protocol reports.

## D-011: preregistered Stage0 `torch.compile` A/B acceptance

- The Section 14.1 20-step A/B uses one fixed real primary-train crop-192 effective batch of eight, identical parent weights, seeds, optimizer/scheduler, BF16/TF32 settings, and optimizer steps 0–19. Both modes run 20 optimizer steps; step 0 is excluded from steady-state throughput because the compiled backward graph is created there.
- Numerical consistency is accepted only if all values are finite, initial and post-step-20 output max/mean absolute differences are at most `2e-3` / `1e-5`, maximum/mean per-step loss differences are at most `1e-5` / `2e-6`, and final model-parameter max/mean absolute differences are at most `5e-5` / `1e-7`.
- Compile is recommended only when every numerical gate passes and steady-state images/s is at least `1.05x` eager. Compilation failure, OOM, a numerical-gate failure, or a smaller speedup deterministically retains eager mode; such a safe disabled result completes the required A/B and is not a Stage0 failure.
- The result is bound to the config, parent checkpoint, PyTorch/CUDA/GPU identity, crop, batch, and compiler options in `artifacts/audits/stage0_compile_ab.json`. It is measured before the 100-step integration and cannot be changed after formal step 0.

## D-012: Stage2 absent-skill effect-profile guard — `DETERMINISTIC_INTERPRETATION`

- Contract omission: Section 8.1 requires executing every skill once on each
  single-degradation sample, but Stage1 has no Planner-generated guard and an
  absent skill has no ground-truth spatial support.
- Resolution: the present source skill uses its continuous ground-truth guard.
  When profiling a different, absent forced skill, force its execution presence
  to one and use a full-support unit guard. A zero guard is forbidden because it
  would record an identity operation while claiming that the skill executed;
  borrowing the source skill's guard would fabricate absent-skill localization.
- Scope and audit: this policy is used only for the 56 absent source-to-forced
  buckets in the single-degradation Stage2 effect profile. Every record and the
  aggregate artifact persist the guard-policy string; Group B/C data, relation
  labels, formal inference, and Stage4 predicted-guard counterfactuals are not
  affected.
- Scientific impact: this defines the missing intervention support needed to
  make “execute every skill once” operational. It is a deterministic project
  interpretation, not AgenticIR protocol reproduction.

## D-013: Stage0 variable-shape validation allocator — `ACCURACY_NEUTRAL_ENGINEERING_RECOVERY`

- Trigger: the first formal step-4000 validation computed all 1600 locked
  `primary_val` images, then failed closed because CUDA peak reserved memory was
  `0.9264`, above the unchanged `0.9000` ceiling. No metric artifact,
  calibration row, best checkpoint, or validation commit was published. The
  raw resumable checkpoint remains at step 4000 with
  `pending_validation_step=4000` (SHA256
  `259af613637e0bfd78a88935aa8de3b96c610c96a4db21048d2a0124cbecc79f`).
- Root cause: the 1600 full-resolution images contain 89 shapes. Repeated
  variable-shape Restormer inference plus the locked FP64 CUDA SSIM convolutions
  accumulated inactive native-allocator segments/cuDNN workspaces. The isolated
  maximum 2040x2040 validation probe was only `43.9236%`; model, EMA, and Adam
  state together account for only about 388 MiB and cannot explain the sequence
  peak.
- Recovery: the resumed process sets exactly
  `PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`. The 90%
  ceiling, native allocator backend, model, weights, optimizer, RNG, data order,
  BF16/TF32/cudnn-benchmark settings, official quantization, PSNR/SSIM formula,
  aggregation, and checkpoint provenance remain unchanged. No
  `max_split_size_mb`, garbage-collection override, `cudaMallocAsync`, metric
  downsampling, tiling, skipped sample, or relaxed guard is used.
- Empirical gate: with the complete step-4000 training state resident, a
  formal-like 100-image prefix containing the maximum image reduced peak
  reserved memory from `17,540,579,328` to `9,758,048,256` bytes (`69.434%` to
  `38.627%`) and inactive-split peak from `4,284,498,432` to zero. In a
  controlled deterministic 100-image causal comparison, all 100 prediction
  hashes and every PSNR/SSIM value were exactly equal while peak reserved memory
  fell from `24,448,598,016` to `10,766,778,368` bytes (`96.780%` to `42.620%`).
  Default-vs-default `cudnn.benchmark` reruns showed larger cross-process PSNR
  jitter than default-vs-expandable, so the small uncontrolled rerun jitter is
  not allocator-specific.
- Evidence: `artifacts/audits/stage0_validation_allocator/` contains the probe
  (SHA256 `9ef0280e...a408`) and all five raw JSON receipts. The resumed run must
  replay the pending validation before step 4001; the complete 1600-image replay
  remains the authoritative runtime memory gate.
- Outcome: the pending replay completed all 1600 images and committed with
  internal peak reserved memory `11,773,411,328` bytes (`46.6050%`). The metric
  artifact records `replayed_pending=true`; `best_ema.pth` and raw `last.pth`
  passed role/state audits, pending was cleared, and training resumed at step
  4001. Thus the full runtime gate—not merely the prefix probe—accepted the
  correction.
- Scientific impact: none. This changes only how unused native CUDA blocks are
  represented/reused; it does not change the research design or numerical
  protocol.

## D-014: Stage0 training-SSIM FP32 correction — `CONTRACT_CORRECTNESS_RECOVERY`

- Trigger: after the locked SSIM weight became active, 17 of 25 periodic
  observations through transient step 12500 had a negative `1-SSIM` term
  (minimum `-0.555595`). The outer BF16 autocast region recast the five SSIM
  convolutions to BF16 even though their inputs had been converted to FP32;
  cancellation in `E[x^2]-E[x]^2` then produced impossible SSIM values above
  one. The process was intentionally stopped before these updates reached a
  checkpoint.
- Contract correction: `train_ssim_y` now disables autocast internally and
  performs Y conversion and all SSIM moments in FP32. It uses the algebraically
  equivalent difference form for the luminance and contrast factors plus
  centered moments, avoiding cancellation without clamping the final SSIM or
  severing gradients. Official quantized PSNR/SSIM functions are unchanged.
- Safe boundary: the authoritative raw checkpoint is step 12000, produced by
  optimizer input step 11999. All 602 recorded formal training points through
  that boundary have `lambda_ssim=0` and `ssim_loss=0`; the broken function did
  not participate in the accepted formal weight trajectory. The later
  step-12020…12500 log rows are classified as discarded transient work and are
  not represented in model, EMA, optimizer, scheduler, RNG, or sampler state.
- Revalidation: full regression is `137 passed`; Ruff and syntax checks pass;
  official metric parity remains PSNR max error zero and SSIM max error
  `3.87908e-7`. Fresh maximum-size validation, compile A/B, CUDA one-batch, and
  exact-100 gates pass. The corrected worst-phase probe retains crop192,
  micro4, accumulation2, eager: `15.588 img/s`, peak `44.9945%`; exact-100 is
  finite at `20.2297 img/s`, peak `31.7701%`.
- Provenance migration: the old checkpoint provenance hash
  `25c56dfd...75187` changed to `aa38a917...59200`. A fail-closed migration
  accepted exactly three SHA leaves: the semantic training-metric source, the
  same source in compile-A/B bindings, and the freshly regenerated compile-A/B
  artifact. Every non-provenance checkpoint section was recursively verified
  bit-exact after serialization. Normal checkpoint provenance verification is
  still mandatory; there is no ignore/bypass flag.
- Recovery artifacts: the original raw/best files remain available as hard
  links with SHA256 `4ef4c817...304f8` and `2c68e142...901a8`. The migrated
  raw/best SHA256 values are `ed2da98d...5ef48` and `4e7d3b0d...71ab3`, with
  full receipts under `artifacts/audits/stage0_ssim_fp32_migration_{last,best}.json`.
- Scientific impact: this restores the V7.1 objective
  `L_ssim = 1 - SSIM_train_Y`; it is a correctness repair, not a new loss,
  protocol relaxation, or additional scientific design choice.

## D-015: Stage1 frozen-scope EMA correction — `CONTRACT_CORRECTNESS_RECOVERY`

- Trigger: the first Stage1 attempt reached a raw/resumable step-3000
  pre-validation checkpoint while the V7.1 phase-0 raw backbone remained
  correctly frozen and bit-exact to the Stage0 parent. The generic EMA still
  applied `decay*x + (1-decay)*x` to those unchanged FP32 tensors. Repeated
  rounding changed 479/495 shared EMA tensors and 5,450,213/25,437,220
  elements; the maximum absolute error was `1.860857e-4`. After BF16
  projection, 82,308 elements still differed, so the pending validation would
  not have represented the contractually fixed Stage0 backbone.
- Fail-closed boundary: the process was interrupted at step 3000 after the
  durable pending checkpoint and before validation publication. The interrupt
  event records `mid_optimizer_update=false`; no Stage1 metric, best
  checkpoint, report, completion event, or calibration row was committed. The
  rejected attempt is preserved under
  `artifacts/archives/stage1_rejected_ema_scope_20260815T155215Z/`, whose
  immutable archive receipt has SHA256
  `623f562123aa3e89bcb14cb7ce648ac20e0ce80e2d4a98185bd5598ae1f5f944`.
- Contract correction: Stage1 now uses a dedicated phase-aware EMA. At each
  optimizer update, parameters whose current `requires_grad` is true receive
  the unchanged standard FP32 EMA formula; currently frozen parameters and all
  buffers use exact `copy_`. The zero-based phase boundary is explicit:
  internal steps 0--4999 keep the Stage0 backbone exact, while internal step
  5000 (logged/checkpointed as step 5001) performs the first ordinary EMA
  update after unfreezing, without resetting the shadow. The formal trainer
  and the worst-phase micro-batch probe use the same implementation.
- Resume hardening: every Stage1 checkpoint binds the EMA scope/policy, decay,
  update count, shape and dtype. Resume validates the complete checkpoint
  contract before mutating model, EMA, optimizer, scheduler, sampler, or RNG.
  A deterministic `optimizer_state_name_ledger` records exactly the sparse
  serialized Adam states that actually exist. State IDs, ledger IDs, current
  parameter names, phase roles, tensor states and phase-local Adam-step bounds
  must agree. This rejects missing/cleared/deleted state without incorrectly
  requiring inactive teacher-forced skills to have Adam history. Stage2 and
  Stage3 also reject legacy or forged Stage1 EMA scopes before loading a
  parent.
- Revalidation: affected tests pass `65/65`; the complete CPU suite passes
  `183/183`; Ruff, `git diff --check`, and compileall pass. Independent
  counterexamples verified pre-mutation rejection for a cleared or partially
  deleted optimizer state, missing/wrong ledger, wrong ID/name, forbidden
  phase role, phase-local step overflow, malformed Adam tensors, scheduler/LR,
  sampler, metric, provenance, EMA, and CUDA-RNG drift. Legal sparse step-1,
  step-0 empty, and step-5000-to-5001 boundary restores pass.
- Code identity: the correction is commit
  `75ec9d44c518fd1a9326989665468c358e1122e8`. The final Stage0 best remains
  unchanged at SHA256
  `52a8744582e39e4f1aa052cc84924ad486289c0b97fc30c89fc6489e69dfac8a`.
- Recovery rule: the rejected Stage1 attempt will not be resumed or migrated.
  A future authorized restart must be a fresh Stage1 step-0 child, loading the
  unchanged final Stage0 best and generating a new run contract/provenance,
  micro-batch probe, and step-0 anchor. As of this record, Stage1 has not been
  restarted; tmux is absent and the GPU is released pending user review.
- Scientific impact: this restores the already locked phase-0 meaning of
  “Stage0 backbone frozen.” It changes neither data, architecture, loss,
  optimizer hyperparameters, schedule, budget, metric, nor checkpoint-selection
  rule.

## D-016: Stage2 pause-status canonical field aliases — `CONTROL_DOCUMENT_CORRECTION`

- Trigger: the completed Stage2 state, approval marker, process exit and GPU
  release were semantically correct, but the generated `RUNNING_STATUS.md`
  used lowercase `gpu` and `next_command` rather than all five literal field
  names required by V7.1 §8.5.3. It also omitted explicit `Stage3: NOT
  STARTED` and `waiting_for: user approval` lines.
- Correction: the current pause document was amended with the four missing
  canonical aliases while retaining the existing diagnostic fields. No source
  code, config, manifest, checkpoint, relation label, decision, orchestration
  state or approval marker was modified.
- Frozen-chain safety: `RUNNING_STATUS.md` is not one of the 22 Stage3 approval
  bindings. After the correction, all 22/22 physical hashes remain exact;
  `stage2_decision.json` and `STAGE3_APPROVAL_REQUIRED.json` remain at SHA256
  `434e209ac0db201ca7f1be045e3811547d8f3cc974ff3ef740c96c3689329a47`
  and `33be4aba2c4229175ac33edef7a5914a48a249b8c733d86338c64a8662072825`.
- Result: the five-line canonical pause block is exact and the new status-file
  SHA256 is `5c5f1c101d244ae26fd36f1bd65cc2fdcfdbb860c9d166586787cc8a621ee9e3`.
  `STAGE3_APPROVED.json` remains absent; production Stage3 still rejects before
  CUDA initialization, and GPU/process/tmux state remains released/empty.
- Scope rationale: the renderer itself was intentionally left unchanged at
  this approval boundary. Editing it would alter a future Stage3/4 semantic
  source hash without changing model behavior; the current document is not
  subject to a background writer and is sufficient for the mandated pause.

## D-017: blur guard uses official ordinal-severity normalization — `RETROSPECTIVE_PROTOCOL_DISCLOSURE`

- Trigger: the required operator/guard protocol reports were missing when the
  completed Stage0/Stage1 lineage started. The approval-boundary audit rebuilt
  them from the locked AgenticIR source and found that operator parameters and
  pixels are exact, while the global guard implementation maps both motion and
  defocus blur as `severity/2`.
- Exact distinction: AgenticIR exposes three ordered tuples. Motion uses
  `(radius,sigma)={(10,3),(15,5),(15,8)}` and defocus uses
  `(radius,alias_blur)={(3,.1),(4,.5),(6,.5)}`. The frozen implementation maps
  their official ordinal control `{0,1,2}` to `{0,.5,1}`. This is tuple-level
  min–max normalization, but it is not literal radius min–max for defocus,
  whose middle value would be `1/3`; nor is it an independently combined
  motion-radius/sigma formula.
- Lineage rule: no post-hoc code or checkpoint mutation is permitted. Stage1
  consumed the ordinal mapping as its teacher-forced guard input, so silently
  changing it now would mix two guard protocols under one parent SHA. The
  current completed lineage therefore remains frozen and both reports state
  the actual mapping exactly.
- Scope: official degradation output, recipe seeds/order, subset targets,
  restoration metrics, Stage0 training, Stage2 relation labels, and MiO100
  boundaries are unchanged. The distinction affects the scalar guard value for
  the middle motion/defocus severity during guarded Stage1 and downstream guard
  supervision.
- Approval consequence: this disclosure does not authorize Stage3. If V7.1
  §2.4.2 is interpreted as requiring literal parameter-wise radius/sigma
  min–max rather than min–max of the official discrete severity control, the
  existing Stage1 lineage is not eligible for that stricter claim and must be
  retrained from its Stage0 parent. User adjudication is required before any
  Stage3 approval; the implementation will not hide the distinction or
  retroactively relabel the completed checkpoint.

## D-018: Stage3 threshold-calibration padding — `ENGINEERING_CORRECTNESS_FIX`

- Trigger: the frozen step-12000 selected-best validation completed all 1600
  `primary_val` images, but the subsequent presence-only calibration called
  `model.encode(image)` on the unpadded full-resolution tensor. The first
  `1356x2040` image therefore reached a `PixelUnshuffle` at height 339 and was
  rejected. This happened after the original selected validation was
  atomically published and before thresholds or `complete.json` existed.
- Correction: the calibration path now calls the same canonical
  `pad_to_multiple(image, 8)` utility used by formal `GraphRestore.forward`.
  The padded tensor is shared by `x0`, `xt`, trace/planner inputs and Encoder
  inputs; there is no resize, edge discard, alternate canonicalization or new
  data materialization. Presence uses the global planner output. The existing
  right/bottom guard-alignment rule remains unchanged.
- Targeted evidence: `test_stage3_threshold_calibration_padding` covers the
  four aligned/non-aligned height/width combinations and one physical
  non-8-divisible `primary_val` image. It checks common utility identity,
  finite `1x8` presence output, manual-prepad equality, and exact
  `crop_to_shape(padded, original_shape) == input`. The frozen selected
  checkpoint SHA is checked before and after the physical-data case.
- Frozen scientific state: `best_ema.pth`, the original
  `selected_validation.json`, the six-row Stage3 calibration history, config,
  optimizer/scheduler/EMA state, checkpoint ranking and the 12,000-step
  selected parent are not modified. The dedicated finalizer creates no
  optimizer, scheduler or train loader and cannot write a checkpoint.
- Scientific impact: none. This repairs support for already-valid image sizes
  in the one-time post-selection diagnostic path; it changes neither weights
  nor the V7.1 training, metric, threshold-search or selection protocol.

## D-019: withdrawn Stage3 extension — `AUTHORIZATION_REVOCATION_AND_AUDIT`

- Trigger: an earlier user message authorized three additional Stage3
  validation cycles, but the subsequent final adjudication explicitly
  superseded it and froze step 12000 as the only Stage3 parent. The extension
  process was stopped and GPU resources were released before any 14,000-step
  validation result, best checkpoint, selection history row, threshold or
  Stage4 artifact was committed.
- Disposition: the transient raw checkpoint at step 14000 has
  `pending_validation_step=14000` and is permanently classified as
  `abandoned_unselected_extension_state`. It is retained byte-exact only for
  audit and is never loaded by the finalizer or Stage4. The selected
  `best_ema.pth` remains the step-12000 SHA
  `9114974f68f202119d4241077d0c46333315204959d58b7eabecaf68a3e32ff3`.
- Audit archive: the extension run contract, abandoned raw checkpoint, train
  log, failed orchestration state, historical extension authorization and
  console were copied to independent mode-0444 inodes under
  `artifacts/migrations/stage3_extension_revoked_after_14000_v1/`. The archive
  receipt SHA is
  `3d92096db631049028ae15f00fc8800edebad3f9e2831ed110fed059ddf02995`.
- Permanent control: canonical `STAGE3_EXTENSION_REVOKED.json` is a tombstone
  checked before any CUDA query, optimizer, scheduler or training loader.
  Plain and extension Stage3 resumes are both refused; the only authorized
  path is step-12000 finalize-only followed by the already-approved Stage4.
- Scientific impact: the extra 2,000 transient optimizer steps and their
  pending raw state are excluded from every reported result and downstream
  parent. They do not alter the frozen step-12000 checkpoint, original selected
  validation, six formal validation rows, threshold calibration or Stage4
  initialization.

## D-020: Stage4 stage-specific calibration ledger — `ENGINEERING_CORRECTNESS_FIX`

- Trigger and contract conflict: V7.1 Sections 12.3 and 17.1 name
  `artifacts/metrics/calibration_history.csv` as the validation ledger, but the
  later authoritative Stage3 revocation/finalization binds that file byte-exact
  at SHA256
  `b282987c3f77034f76788a412e91823cd4570ce8c6c10cd93030ee181612e034`.
  Appending Stage4 rows would invalidate the frozen Stage3 completion evidence.
  The first Stage4 step-4000 validation completed all 1,600 images, then the
  shared-ledger writer also classified legitimate historical Stage0/Stage3
  step-4000 rows as duplicate Stage4 rows and failed closed before clearing the
  pending transaction or permitting step 4001.
- Correction: preserve the shared Stage0/Stage3 ledger byte-exact and route
  only Stage4 validations to the sibling
  `artifacts/metrics/stage4_calibration_history.csv`, schema
  `graphrestore-stage4-calibration-ledger-v1`. The sidecar keeps the same
  locked 28 columns and exact validation schedule `4000..40000`; six
  Stage4-only misuse fields identify Stage4 rows.
- Transaction guards: Stage4 requires a contiguous unique schedule prefix,
  atomic/idempotent append, exact-row replay, and strict
  path/header/type/symlink/hardlink/hash checks. The frozen shared SHA is
  checked before CUDA use and across every append; routing and both ledgers are
  bound into the Stage4 contract/checkpoints and final report/completion
  evidence.
- Provenance: the controlled routing migration and follow-up
  finalization-binding source fix are COMPLETE (receipt SHA256
  `795982a5f607c147e25a2553a63c4b24306fd0fe2753cdcd6ca0cab0af8c190d` /
  `3be807f84ec1b4d12141bfb4de75040b0124706eefe863eeabfebfa2c140bdcd`;
  routing SHA256
  `6fa3a7f6eb6c5ad3790ed7ea2d332c9d422e3999f2adc7bfa47495830e3802a0`).
  Both Stage4 checkpoints are bit-exact outside provenance; model, EMA,
  optimizer, scheduler, sampler, RNG and pending step remain unchanged.
- Scientific impact: none. This is a stage-ownership/path correction only. It
  changes no model computation, training sample/order/budget, validation
  images/seeds, metric definition/aggregation/value, threshold, diagnostic, or
  restoration-first checkpoint-selection rule, and does not authorize
  MiO100/Group-B/Group-C access.

## D-021: AgenticIR Table-1 cross-backend finalization — `EVIDENCE_RECOVERY`

- Trigger: formal inference completed all 1,440 images and the pinned
  AgenticIR/pyiqa CUDA scorer durably committed all 144 ten-image score shards.
  The scorer then failed closed before publishing its aggregate files because
  an auxiliary crosscheck treated the observed maximum error from a 24-pair
  CPU/CPU parity sample as a universal CPU/CUDA per-image tolerance.
- Diagnosis: prediction and target identities, paths and SHA256 values matched
  exactly for all 1,440 rows, and all six official-CUDA metrics were finite.
  The only differences were ordinary cross-backend floating-point drift between
  the CPU fast diagnostic and the official CUDA scorer: maximum PSNR drift
  `3.814697265625e-6 dB` and maximum SSIM drift
  `1.934401321468382e-5`. The parity artifact's observed sample maxima were
  never a theoretical cross-device bound.
- Recovery: a separate immutable approval authorized a CPU-only finalizer to
  verify the complete shard/RNG/evidence chain and publish aggregates directly
  from the already committed CUDA shards. It did not decode an image, launch a
  metric model, initialize CUDA, rerun a sample, change a tolerance, or alter a
  score. The official CUDA shards remain the sole Table-1 metric authority;
  the CPU values are retained only as an identity-bound drift diagnostic.
- Terminal evidence: the recovery approval SHA256 is
  `e6b73613dce9772a17d2963608ce67b42485ef7a2eec4cc09b42a4a42fdaae7c`;
  the terminal receipt SHA256 is
  `fc3aec42a22f06fe866adf988aa2a25ddb591bf7ef54ef1868f9c550cecb7ce9`.
  The receipt records `numeric_parity_claim=false`,
  `numeric_gate_applied=false`, and `tolerance_changed=false`.
- Scientific impact: none on predictions or metric values. This corrects the
  role of an auxiliary diagnostic and preserves the original one-shot official
  CUDA scoring output without a result-driven rerun.
