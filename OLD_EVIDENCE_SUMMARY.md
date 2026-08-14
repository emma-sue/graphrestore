# Old Evidence Summary

Source absorbed from `/root/autodl-tmp/second_ir_work/one_click_second_ir/OLD_EVIDENCE_SUMMARY.md`; this file records why GraphRestore must remain orthogonal to FreqLock and earlier calibration routes.

## FreqLock boundary

FreqLock already covers frequency locking, low-frequency structure preservation, latent diffusion, perceptual quality, and downstream classification/segmentation consistency. GraphRestore therefore cannot use frequency, low/high-frequency experts, task-driven downstream objectives, or UniRestore-style prompt routing as its main story.

## Prior attempts

- Frequency-band/adaptive step showed a dehaze oracle but was unstable for denoise/derain.
- MEB multi-exit gains were task-dependent; Exit Head was weak and denoise did not improve reliably.
- RGM plus soup/TTA produced a five-setting average gain around +0.3111 dB, but it is an already-explored PromptIR/MEB tail-gain route.
- R2R all-bank soft retrieval catastrophically collapsed several tasks to 4–5 dB; conservative output calibration recovered only tiny gains (maximum old full delta about +0.036888 dB).

## Retained evidence

- Over-restoration and negative transfer are real.
- Wrong control signals can cause catastrophic interference.
- Denoising is especially sensitive to structural/output perturbation.
- Per-task, per-image, and worst-drop diagnostics are required.

## Reusable assets and strict exclusions

Reusable only as diagnostic patterns: reliability logging, exact-baseline switches, per-image CSVs, and fair ablation discipline. The current V7.1 project does not load R2R/ProVIR skill weights or old verifier/memory modules.

Never repackage frequency-aware planning, MEB/adaptive exits, RGM/tail gain, output-only guard, alpha scan, task-level gain/bias, TTA, or model soup as the GraphRestore contribution.
