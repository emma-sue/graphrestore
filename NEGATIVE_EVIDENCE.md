# Negative Evidence

- Existing OpenCV-upsampled MiO100 canonical files are invalid as V7.1 formal model inputs; online BasicSR canonicalization from native LQ is mandatory.
- RAR broad data are incomplete and prohibited by the frozen data contract.
- ProVIR verifier/memory/DINO/EOA/EAR/State-K2 components and weights are prohibited for GraphRestore.
- OPERA implementation, training, metric, and data details are outside scope and must not be read or used.
- `training_data/scripts/materialize_primary.py` cannot be reused as the training Dataset: its haze depth adapter returns inconsistent roots on create vs reuse (eventually forming `<id>/<id>/predict_depth.mat`), does not preserve/restore all RNG states, and uses `lr(..., keep_size=True)`. The new adapter must prebuild the full compatibility tree and retain native 1/4 uint8 before BasicSR canonicalization.
- The old `lr(..., keep_size=True)` preview path is measurably non-equivalent to V7.1 native-uint8 then BasicSR float ×4: one audited 2040×1356 recipe had float `max_abs=0.01333`, `mean_abs=0.001114`; after uint8 quantization, 1,484,065 / 8,298,720 channel values differed (max 3). Existing previews are never formal model inputs.
- Old R2R all-bank soft retrieval collapsed several tasks to 4–5 dB; unguarded linear bank mixtures are prohibited evidence, not a reusable design.
- Old conservative R2R output calibration peaked around +0.036888 dB and is below a meaningful architecture gain; output gain/bias/alpha is not a contribution.
- MEB/adaptive exits, frequency-band steps, RGM/tail modulation, TTA and model soup are prior routes or training aids and cannot replace the three frozen GraphRestore contributions.
- A generic EMA over nominally frozen FP32 Stage1 backbone tensors is invalid: repeated identity mixing changed 479/495 shared shadows and survived BF16 projection in 82,308 elements. The rejected run was stopped before its first validation commit and must never be resumed or reported.
- Exact scheduler validation must mirror the production floating-point operation order. Algebraically equivalent warmup expressions differed by one ULP at fresh Stage1 step0; the failed pre-anchor initialization produced no metric/checkpoint and is archived, not a training result.
- Stage2 does not support collapsing the design to either a global total order or unconditional parallel fusion: val non-ambiguous parallel share is 0.6476, while the serial-gap median/P75 is 0.1998/1.1097 dB and pair-majority labels are context-dependent. The evidence supports retaining a sample-conditioned partial-order planner.
