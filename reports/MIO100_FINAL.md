# Formal MiO100 Group A/B/C Evaluation

## Status

- Status: **VERIFIED / COMPLETE**
- Protocol: `graphrestore-v7.1-agenticir-locked`
- Method label: `GraphRestore v7.1 Stage4 step-40000`
- Selected checkpoint: `artifacts/checkpoints/stage4/best_ema.pth`
- Checkpoint role: `ema_selection`, `resumable=false`, `step=40000`
- Checkpoint SHA256: `6aa2de6e65ce633430d188857845acef1b67cfb3b218a04977293dbc149a84fd`
- Evaluation set: frozen online-canonical MiO100 test, 1,440 images total
- Groups: A=`8 x 80 = 640`, B=`4 x 100 = 400`, C=`4 x 100 = 400`
- Inference: autonomous planner/compiler, no task-label routing, no TTA, no model soup, no threshold tuning, and no result-driven rerun

## Paper-style six-metric result

The values below use the AgenticIR/RAR column order. PSNR and SSIM are
full-reference metrics; LPIPS is lower-is-better; MANIQA, CLIP-IQA, and MUSIQ
are higher-is-better. GraphRestore values are recomputed group means from the
frozen official-CUDA per-image score shards. Published comparison values are
from the [AgenticIR ICLR 2025 paper](https://openreview.net/pdf?id=3RLxccFPHz)
and the [RAR CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.pdf).
Bold marks the best value among only these three rows within each group; it is
not a claim against every method in the papers.

| Group | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MANIQA ↑ | CLIP-IQA ↑ | MUSIQ ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| A | **GraphRestore v7.1** | **24.37** | **0.7599** | 0.2732 | 0.2742 | 0.4127 | 53.93 |
| A | AgenticIR (published) | 21.04 | 0.6818 | 0.3148 | 0.3071 | 0.4474 | 56.88 |
| A | RAR (published) | 20.46 | 0.7144 | **0.1299** | **0.4659** | **0.6566** | **57.19** |
| B | GraphRestore v7.1 | 19.82 | 0.6679 | 0.3619 | 0.2727 | 0.3968 | 50.53 |
| B | AgenticIR (published) | 20.55 | 0.7009 | 0.3072 | 0.3204 | 0.4648 | **57.57** |
| B | RAR (published) | **21.04** | **0.7326** | **0.1269** | **0.4582** | **0.6483** | 56.91 |
| C | GraphRestore v7.1 | 18.52 | 0.5563 | 0.5059 | 0.1971 | 0.2751 | 38.06 |
| C | AgenticIR (published) | 18.82 | 0.5474 | 0.4493 | 0.2698 | 0.3948 | 48.68 |
| C | RAR (published) | **19.33** | **0.6579** | **0.1489** | **0.4653** | **0.6554** | **56.56** |

### Full-precision GraphRestore group means

| Group | Images | Combinations | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MANIQA ↑ | CLIP-IQA ↑ | MUSIQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 640 | 8 | 24.3741960555 | 0.7599264849 | 0.2731640404 | 0.2741912930 | 0.4126837323 | 53.9287128568 |
| B | 400 | 4 | 19.8229607308 | 0.6678735027 | 0.3619331664 | 0.2726837824 | 0.3968436596 | 50.5320779753 |
| C | 400 | 4 | 18.5247367167 | 0.5563379351 | 0.5058757574 | 0.1970546311 | 0.2750860956 | 38.0631561780 |

## Per-combination results

Each Group-A combination contains 80 images; every Group-B/C combination
contains 100. Group values above are equal-weight means of the combination
means, matching the frozen AgenticIR aggregation protocol.

| Group | Combination | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MANIQA ↑ | CLIP-IQA ↑ | MUSIQ ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| A | rain + haze | 25.2091 | 0.897448 | 0.094386 | 0.418710 | 0.616096 | 66.5928 |
| A | motion blur + low resolution | 21.7352 | 0.613996 | 0.417192 | 0.189613 | 0.283185 | 38.3954 |
| A | dark + noise | 26.3337 | 0.852018 | 0.160268 | 0.342081 | 0.571815 | 65.1134 |
| A | defocus blur + JPEG | 25.1875 | 0.713920 | 0.398806 | 0.213730 | 0.249878 | 45.3816 |
| A | noise + JPEG | 28.3028 | 0.843963 | 0.218073 | 0.298527 | 0.454993 | 61.5533 |
| A | rain + low resolution | 25.3412 | 0.734568 | 0.332016 | 0.273335 | 0.346511 | 52.3587 |
| A | motion blur + dark | 20.4950 | 0.645442 | 0.335530 | 0.171317 | 0.299373 | 44.2693 |
| A | defocus blur + haze | 22.3891 | 0.778057 | 0.229042 | 0.286217 | 0.479619 | 57.7653 |
| B | motion blur + JPEG | 21.7529 | 0.644968 | 0.384693 | 0.181721 | 0.291449 | 44.5297 |
| B | haze + noise | 14.2062 | 0.645195 | 0.318303 | 0.295390 | 0.495650 | 58.3561 |
| B | defocus blur + low resolution | 24.7119 | 0.677713 | 0.454644 | 0.201064 | 0.267164 | 34.2283 |
| B | rain + dark | 18.6209 | 0.703618 | 0.290092 | 0.412559 | 0.533112 | 65.0142 |
| C | haze + motion blur + low resolution | 15.4956 | 0.515951 | 0.500503 | 0.183200 | 0.300553 | 33.8743 |
| C | rain + noise + low resolution | 23.5156 | 0.652399 | 0.506178 | 0.248411 | 0.274932 | 47.6937 |
| C | dark + defocus blur + JPEG | 15.6997 | 0.522538 | 0.463476 | 0.210507 | 0.244687 | 43.4249 |
| C | motion blur + defocus blur + noise | 19.3880 | 0.534464 | 0.553346 | 0.146101 | 0.280172 | 27.2598 |

## Scientific reading

- **Group A is the clear strength.** Relative to published AgenticIR,
  GraphRestore gains `+3.3342 dB` PSNR and `+0.07813` SSIM while also reducing
  LPIPS by `0.04164`. Relative to published RAR, it gains `+3.9142 dB` and
  `+0.04553` SSIM. The no-reference IQA scores, however, remain below both
  AgenticIR and RAR.
- **Group B does not generalize competitively.** Relative to AgenticIR it is
  `-0.7270 dB / -0.03303 SSIM`, and all four perceptual/no-reference metrics
  are also worse.
- **Group C is mixed but overall weak.** SSIM is `+0.00894` above published
  AgenticIR, but PSNR is `-0.2953 dB`; LPIPS and all three no-reference IQA
  scores are worse. It trails RAR on all six metrics.
- Therefore this run supports a claim of strong restoration fidelity on the
  familiar Group-A mixtures, but **not** a blanket SOTA claim and **not** strong
  unseen-composition/triple-degradation generalization. The B/C gap is the main
  scientific failure mode to address next.

## Evaluation integrity and controlled recovery

- Formal inference finished all 1,440 images once, with 1,440 immutable image
  receipts and zero missing, duplicate, or pending samples. The prediction
  digest is `e922e274f42d40a93673252336950595f1bb552e83d0abf75992e40424ffce6e`.
- The six-metric scorer completed 144 immutable CUDA shards (10 images each).
  Every metric was finite; maximum reserved VRAM was `0.150009469`, below the
  locked `0.90` ceiling.
- The original scorer then failed closed during an auxiliary cross-backend
  check. It incorrectly treated the observed maximum from a 24-pair CPU/CPU
  parity sample as a universal CPU/CUDA tolerance. The 1,440 prediction and GT
  hashes were exact; only normal backend-level floating-point differences were
  present (maximum PSNR drift `3.8147e-6 dB`, maximum SSIM drift `1.9344e-5`).
- A separate immutable approval and CPU-only recovery finalizer published the
  already completed official-CUDA scores. It did not read images, launch a
  metric worker, initialize CUDA, change a tolerance, or recompute any metric.
  The scientific source of truth remains the original 144 official CUDA
  shards. `numeric_parity_claim=false`, `numeric_gate_applied=false`, and
  `tolerance_changed=false` are recorded in the terminal evidence.

## Primary artifacts

- Formal inference complete: `artifacts/formal_mio100/graphrestore_v7_1_stage4_step040000/complete.json`, SHA256 `19efefd629a03bbbf85005dc728b93d8f6d8edd04bbfcb8bd8b8c19ae958e000`
- Six-metric per-image CSV: `artifacts/formal_mio100/graphrestore_v7_1_stage4_step040000/table1_scores/per_image.csv`, SHA256 `f8bcbd463eb7113ccf632b5e03f0a34951650a8b0e66e1e63edabbfdd781a6d5`
- Six-metric summary: `artifacts/formal_mio100/graphrestore_v7_1_stage4_step040000/table1_scores/summary.json`, SHA256 `68301ea9f0ef52f062a8fdaa763865c3fb0374ee49f65ec9d99cdd9cf5ac7be8`
- Six-metric complete: `artifacts/formal_mio100/graphrestore_v7_1_stage4_step040000/table1_scores/complete.json`, SHA256 `eb1468705d2591709565b3461aa630ae826f1efc311cddaf6bc06feca309a60b`
- Recovery approval: `artifacts/approvals/AGENTICIR_TABLE1_BACKEND_RECOVERY_APPROVED.json`, SHA256 `e6b73613dce9772a17d2963608ce67b42485ef7a2eec4cc09b42a4a42fdaae7c`
- Recovery terminal receipt: `artifacts/migrations/agenticir_table1_backend_recovery/COMPLETE.json`, SHA256 `fc3aec42a22f06fe866adf988aa2a25ddb591bf7ef54ef1868f9c550cecb7ce9`

Independent terminal verification returned `COMPLETE` with 1,440 images, 144
shards, unchanged shard inventory, exact summary/receipt recomputation, no
partial artifacts, no live evaluator/scorer/recovery process, and an empty GPU
compute-process set.
