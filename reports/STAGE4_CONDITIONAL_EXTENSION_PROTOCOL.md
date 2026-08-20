# Stage4 Conditional Extension Protocol

- Protocol: `graphrestore-v7.1-agenticir-locked`
- Authorization mode: one-shot, conditional, fail-closed
- Base training target: `40000` optimizer steps
- Candidate validation: the atomically committed `step=40000` Stage4 validation
- Reference validation: the immediately preceding atomically committed `step=36000` Stage4 validation
- Gate metric: equal-combination mean Group-A PSNR from the Stage4-owned calibration sidecar
- Gate arithmetic: parse the two original UTF-8 CSV field lexemes with Python `Decimal` at precision 80; compute `PSNR_40000 - PSNR_36000` without quantization, rounding, epsilon, or conversion through binary floating point
- Gate condition: `delta >= Decimal("0.20")`
- The checkpoint-selection `<0.02 dB` tie band does not apply to this gate. Single PSNR, Group-A SSIM, and Single SSIM cannot supplement a failed PSNR gate.
- Hard evidence: both validations must contain 1600 primary-val images under the unchanged frozen quantization/metric protocol; all metrics must be finite; the step-40000 raw checkpoint must be resumable with `pending_validation_step=null`; optimizer, EMA, scheduler, sampler, RNG, provenance, metric, report, sidecar, and selected-best bindings must agree.
- If the gate is false: authorize zero additional optimizer steps and let the original Stage4 final diagnostics/completion path continue.
- If any evidence gate is invalid: authorize zero additional optimizer steps and stop fail-closed for audit; this is not recorded as a scientific threshold failure.
- If the gate is true: authorize exactly optimizer steps `40001` through `48000`, with complete validations at `44000` and `48000`.
- The original cosine horizon remains `40000`; every optimizer group remains at its original terminal learning-rate floor after step 40000. Optimizer, EMA, scheduler, sampler, RNG, model topology, data, loss, thresholds, compiler, guard semantics, and selection rule are not reset or changed.
- `48000` is an unconditional hard terminal. No step `48001` and no further result-conditioned extension are authorized.
- Formal MiO100 and Group-B/Group-C reads remain unauthorized and must stay zero.
- The extension, if activated, is an additional 8000 steps (`+20%` training budget) for continued observation only. It does not by itself establish superiority to Stage0, an external baseline, an ablation, or a paper claim.
- The final selected checkpoint remains the best under the frozen Stage4 ordering across every formally committed validation from steps 4000 through 48000; the terminal checkpoint is not forced to win.

This document records the user's instruction to continue after the step-40000 validation only when its immediately adjacent Group-A PSNR gain is at least 0.2 dB, together with the bounded `48000` hard terminal communicated before the gate became observable.
