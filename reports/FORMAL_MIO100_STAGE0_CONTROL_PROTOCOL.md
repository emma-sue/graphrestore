# Frozen Stage0 MiO100 Control Protocol

## Status and authorization gate

- Status: `PREREGISTERED / NOT YET USER-AUTHORIZED`.
- Analysis type: post-Stage4-unblinding frozen baseline control.
- This document does not authorize a MiO100 image read, CUDA inference, metric
  run, or output publication.  A new explicit user instruction and an
  immutable `FORMAL_MIO100_STAGE0_CONTROL_APPROVED.json` are required first.
- Stage4 formal A/B/C results were known before this control was commissioned.
  The Stage0 checkpoint, config and primary-validation selection were frozen
  before any formal MiO100 A/B/C image was read.  Consequently this is a valid
  frozen-control comparison, but it does not restore blind-test status for B/C.

## Scientific question

Does the already-frozen Stage4 GraphRestore checkpoint improve over its
already-frozen Stage0 MiO-StageA scientific anchor under exactly the same
MiO100 input bytes, online canonicalization, persisted-PNG scoring path,
AgenticIR metric implementation and equal-combination aggregation?

## Frozen control and treatment

Control:

- Method name: `mio_stagea_v7_1_stage0_step060000`.
- Output root:
  `/root/autodl-tmp/aaa/graphrestore/artifacts/formal_mio100/mio_stagea_v7_1_stage0_step060000`.
- Checkpoint: `artifacts/checkpoints/stage0/best_ema.pth`.
- Checkpoint SHA256:
  `52a8744582e39e4f1aa052cc84924ad486289c0b97fc30c89fc6489e69dfac8a`.
- Config: `configs/stage0_mio_stagea.yaml`.
- Config SHA256:
  `1bfb0444e311d110c6929ce30fdcf888a73d0f10167ed27e3328492a08406283`.
- Checkpoint identity: `stage0`, step `60000`, `ema_selection`,
  `resumable=false`, `pending_validation_step=null`, 495 model tensors,
  `model == ema.shadow`, EMA updates `60000`.
- Selection source: frozen `primary_val` only, before MiO100 unblinding.

Treatment evidence is immutable and will not be regenerated:

- Stage4 checkpoint SHA256:
  `6aa2de6e65ce633430d188857845acef1b67cfb3b218a04977293dbc149a84fd`.
- Stage4 official-CUDA per-image score SHA256:
  `f8bcbd463eb7113ccf632b5e03f0a34951650a8b0e66e1e63edabbfdd781a6d5`.
- Stage4 official-CUDA summary SHA256:
  `68301ea9f0ef52f062a8fdaa763865c3fb0374ee49f65ec9d99cdd9cf5ac7be8`.
- Stage4 Table-1 complete SHA256:
  `eb1468705d2591709565b3461aa630ae826f1efc311cddaf6bc06feca309a60b`.

## Frozen Stage0 source-compatibility rule

The Stage0 checkpoint records 46 semantic-source leaves from the complete
multi-stage repository at the time it was selected.  Before this protocol was
user-authorized, a CPU-only audit rehashed every leaf and all four Stage0
training/validation manifests.  Forty-one source leaves and all four manifests
remain byte-identical.  Exactly five files changed later while Stage1--Stage4
were implemented or hardened; none enters the Stage0 formal runtime:

| path | checkpoint SHA256 | preregistration SHA256 |
|---|---|---|
| `src/training/orchestration.py` | `597333101407451f83aaede9e6c23be9f222209de007b78a05ce4684f42b0584` | `1aed52f7780b574756e2620092686be55f66ef60619d133cf8265b79439ee66b` |
| `src/training/stage1_engine.py` | `017b031a2424f7bdd2f7481fe223c455607a53e908cef4a37ef007d04c85e961` | `ab76a61422532d0deac8d7c01da69dae9f1d8154a277a4c0c26658b936c36f6c` |
| `src/training/stage2_distillation.py` | `b9c8816b0ad67fbb8ff9c39851b190466307ba44970ec2cf5f69c9085c8be7bf` | `4cd37e4a1e5725e9b948c758eab893ff9a95f1c23357a2cdea3215df93d1b06c` |
| `src/training/stage3_engine.py` | `eecfeecc087b735d085562b26047f99d90160a5cd2938075d1277cf09d9477f5` | `1d373d8d3e416e6431d52721c2cd2eef15541241a2d7c28af8a688181a972548` |
| `src/training/stage4_engine.py` | `7079296d2ab27a09982303c6d609cbba1694966a515474d3b21eeb707b9b669f` | `2cd111a2ed5181fab864784916ead8d889fceaac02fee64f1aae40f50bde0d6c` |

These five files are not imported by the independent Stage0 formal evaluator,
the prompt-free `MiOStageA` model, input canonicalizer, persisted-PNG metric
path, or the official Table-1 scorer.  They are therefore an exact,
pre-authorized compatibility allowlist, not a general provenance bypass.  The
readiness gate must rehash all 46 leaves, require every other leaf to equal its
checkpoint SHA, require the five rows above to equal both their recorded
checkpoint and preregistration SHA values, and require all four manifests to
remain exact.  Any sixth difference or any later change to an allowlisted file
fails closed before approval.

## Frozen data and metric identity

- Online-canonical 1,440-row manifest SHA256:
  `83fb90dfa121681123f55e73df32eb6c1bc37e685c0e27ae07ad7e59a687a7f5`.
- Existing immutable byte inventory SHA256:
  `489d9c216589bb73f4b99ec8301abd57d77b7d418489cef02c162bc135aa91ae`.
  Stage0 must reuse and revalidate this exact inventory rather than create a
  second data identity.
- Metric-weight inventory SHA256:
  `796e39eddc51c28e57b9c40b393f99fd73bd14fde2bab138987c2ddcde746e7d`.
- AgenticIR commit:
  `9640a291480dee3ba8f2974125d4ee9e3440f3d6`.
- AgenticIR scorer SHA256:
  `b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830`.
- Prediction protocol: crop to GT geometry, clamp and round to uint8, persist a
  lossless PNG, then read the persisted PNG for every reported metric.
- Aggregation: arithmetic mean per exact combination, followed by an
  equal-combination arithmetic mean within A, B and C.  The pooled 1,440-image
  mean is supplementary only.
- The official AgenticIR/pyiqa CUDA shards are the only authoritative six-metric
  values.  CPU PSNR/SSIM may be retained only as identity/drift diagnostics.

## One-shot scope

- Exactly one frozen Stage0 checkpoint, one method name and one new output root.
- Inference is prompt-free MiO-StageA and receives only the image tensor.  A
  manifest degradation label may be used after inference for aggregation but
  never as model input or routing information.
- No training, checkpoint selection, threshold/prior use, TTA, ensemble, model
  soup, output selection, architecture change or result-driven rerun.
- No overwrite.  An interrupted run may resume only after exact verification
  of its authorization, contract, immutable PNG and receipt prefix.
- Stage4 artifacts, checkpoint and selected result remain immutable.
- Formal A/B/C inference and Table-1 scoring require an exclusive GPU and peak
  reserved fraction strictly below `0.90`.

## Co-primary endpoints and success rule

For each `G` in `{A, B, C}`:

```text
delta_PSNR_G = PSNR_Stage4_G - PSNR_Stage0_G
delta_SSIM_G = SSIM_Stage4_G - SSIM_Stage0_G
```

Original V7.1 directional gate:

- `delta_PSNR_A/B/C > 0`;
- `delta_SSIM_A/B/C > 0`.

Original V7.1 ideal effect-size target:

- `delta_PSNR_A/B/C >= +0.20 dB`;
- `delta_SSIM_A/B/C > 0`.

These thresholds are frozen before Stage0 inference and will not be revised.

Secondary endpoints:

- Paired PSNR/SSIM deltas for all 16 combinations and all 1,440 images.
- Group/combination LPIPS, MANIQA, CLIP-IQA and MUSIQ deltas.  Oriented LPIPS
  gain is `LPIPS_Stage0 - LPIPS_Stage4`; the other gains are
  `Stage4 - Stage0`.
- Per-image win rates, Stage0 parameter count, latency and peak VRAM.
- A 10,000-resample paired cluster bootstrap by clean ID may be reported only
  with a seed frozen in the authorization.  It describes test-image
  variability, not training-seed uncertainty.

Decision rule:

1. All six directional endpoints positive and every group PSNR gain at least
   `+0.20 dB`: incremental-efficacy gate passes; A1/A3/A4 may be prepared.
2. All directions positive but one or more PSNR gains below `+0.20 dB`:
   intermediate evidence only; run a frozen mechanism audit before expensive
   retraining.
3. Any group PSNR or SSIM delta non-positive: the V7.1 formal target fails; do
   not launch the full A1/A3/A4 retraining suite.
4. A-only gain with B/C non-positive is seen-composition specialization, not
   unseen-composition generalization.

## Required future user authorization

The approval must explicitly authorize only the following bounded action:

> Use the frozen Stage0 step-60000 EMA checkpoint with SHA256
> `52a8744582e39e4f1aa052cc84924ad486289c0b97fc30c89fc6489e69dfac8a`
> once on the same formal MiO100 A/B/C 1,440-image protocol, followed by the
> same six-metric scorer and a frozen Stage4-minus-Stage0 paired comparison.
> This authorizes no training, checkpoint/threshold/prior selection, Stage4
> mutation, TTA, model fusion or result-driven rerun, and it does not restore
> B/C blind-test status.

The immutable approval path is:

`artifacts/approvals/FORMAL_MIO100_STAGE0_CONTROL_APPROVED.json`.

## Claim boundary

This control may establish an exact same-protocol Stage4-versus-Stage0 delta.
It cannot by itself establish a causal contribution for partial ordering,
spatial guards or counterfactual calibration; compute-matched superiority;
training-seed robustness; state of the art; or an untouched blind B/C result.
Same-checkpoint zero-training mode switches are diagnostics, not fair retrained
ablations.  Any later architecture designed from the revealed B/C outcome must
use a separately preregistered OOD development set and a new blind final set.
