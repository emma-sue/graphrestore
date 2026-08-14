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
