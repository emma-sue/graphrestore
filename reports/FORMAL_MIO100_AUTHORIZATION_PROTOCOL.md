# Formal MiO100 A/B/C Authorization Protocol

## User authorization

On 2026-08-20 UTC, the user explicitly authorized evaluation of the final
frozen Stage4 model on the formal MiO100 Group A, B and C test sets, using the
same dataset organization and table-level metrics reported by AgenticIR and
RAR.  This authorization supersedes the earlier `formal_mio100=false` pause
only for the bounded evaluation described below.  It does not authorize any
additional training or model selection.

Exact user instruction:

> 请你把最后训练的这个结果，在group a b c测试集合进行测试，就跟rar agenticir文章里展示的实验结果那种，打印给我我的这个模型在同样的数据集测试的结果

## Frozen scientific scope

- Evaluate exactly one frozen Stage4 `best_ema.pth` selected before any formal
  MiO100 image is read.
- Require Stage4 `complete.json`, the selected-checkpoint binding, the six-mode
  zero-training diagnostics, and GPU release before the formal run starts.
- Read exactly the frozen 1,440-image online-canonical manifest: Group A
  `8 x 80 = 640`, Group B `4 x 100 = 400`, and Group C `4 x 100 = 400`.
- Use autonomous GraphRestore inference with the frozen presence thresholds,
  pair priors, global priority, compiler and maximum three rounds.  Ground
  truth degradation labels may be used only after inference for aggregation
  and diagnostics; they may not control routing, guards, skills or stopping.
- For low-resolution rows, load the official native LQ PNG and apply the
  locked MiOIR/BasicSR x4 float canonicalization online without re-quantizing
  the resized input.  Other rows use the native RGB float input unchanged.
  The manifest's low-resolution flag is permitted only to select this input
  canonicalization path; it must never enter the planner, routing, guard,
  skill-selection or stopping logic.
- Before the first model forward, freeze an immutable byte inventory covering
  every native LQ and GT file referenced by all 1,440 rows.  The inventory and
  its per-row SHA256 bindings must be part of the one-shot authorization and
  must be rechecked during processing, resume and finalization.
- Crop the prediction to the GT geometry, clamp and round to uint8, write a
  lossless PNG, then read the persisted PNG for all reported metrics.
- Report the AgenticIR Table-1 columns: PSNR, SSIM, LPIPS, MANIQA, CLIP-IQA and
  MUSIQ.  Aggregate per image to an arithmetic mean per exact degradation
  combination, then use an equal-combination arithmetic mean within A, B and
  C.  The 1,440-image pooled mean is supplementary only.

## One-shot and fail-closed rules

- No checkpoint swap, threshold/prior change, TTA, ensemble, model soup,
  task-label routing, output selection, or result-driven rerun is authorized.
- The run has a single immutable output root on the data disk and may never
  overwrite an existing completed protocol ID.
- Durable per-image completion may be resumed only by verifying every existing
  PNG and record against the same run contract; resume is recovery, not a new
  scientific trial.
- Missing, duplicate or extra samples; non-finite values; protocol/hash drift;
  VRAM peak at or above 0.90; a competing GPU process; or forbidden data use
  stops the run without silently accepting partial results.
- Metric-model downloads and caches must reside on the data disk.  The locked
  metric environment, metric source files and all downloaded weight files are
  hashed before the six-column table is published.
- The downstream six-metric scorer must revalidate the same authorization and
  the completed formal inference transaction.  It uses one fixed output root,
  requires an exclusive GPU and the same strict VRAM ceiling, and cross-checks
  its per-image PSNR/SSIM against the persisted-PNG inference ledger.

## Claim boundary

The resulting table is a formal internal evaluation of this frozen
GraphRestore checkpoint on the AgenticIR MiO100 A/B/C protocol.  Comparison to
published AgenticIR or RAR numbers is descriptive.  It does not by itself
establish state of the art, isolate the causal contribution of any one module,
or constitute a fair compute-matched ablation.  A formal Stage0 comparison
would require a separate authorization and evaluation of the frozen Stage0
checkpoint under this same protocol.
