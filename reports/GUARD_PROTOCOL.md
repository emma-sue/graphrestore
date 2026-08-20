# GraphRestore Local Skill Guard Protocol

- audit time (UTC): `2026-08-17T06:17:17Z`
- operator source SHA256: `c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c`
- implementation sources: `src/data/agenticir_degradations.py`, `src/data/subset_targets.py`, `src/data/episode_dataset.py`
- value domain: FP32, clamped to `[0,1]`
- output resolution: synchronized full-resolution target followed by `adaptive_avg_pool2d` to exactly `H/4 × W/4`

The target is a local skill necessity/intensity map. It is training-only supervision; it is never computed from ground truth and injected at inference.

## Dense guard targets

| Skill | Exact full-resolution target | Authoritative inputs |
|---|---|---|
| rain | `mean(abs(after_bgr_uint8 - before_bgr_uint8), channels) / 255` | Actual visible, clipped output of the locked rain operator and its immediate input. No binary mask and no rewritten rain equation. |
| haze | `1 - transmission`, where `transmission = exp(-beta * depth / max(full_depth))` | Locked recorded `beta`; MiOIR depth resized x4 using OpenCV `INTER_CUBIC`; normalization is performed on the complete resized depth before any crop. |
| low light | `clamp(1 - Y_after / (Y_before + 1e-6), 0, 1)` | OpenCV BGR→YCrCb Y channel, scaled to `[0,1]`, from the immediate before/after images of the official `dark` operator. |

For dense skills, `global_severity_targets[skill]` is the mean of the resulting dense map. Rain and haze targets remain continuous.

## Global-severity targets

The following values fill the full-resolution guard plane for the present global skill; Stage3 supervises the spatial mean rather than forcing the predicted map to be spatially constant.

| Skill | Frozen formula | Result over official domain |
|---|---|---|
| Gaussian noise | `(sigma - 20) / (50 - 20)` | `[0,1]` |
| Poisson noise | `(scale - 1) / (3 - 1)` | `[0,1]` |
| motion blur | `severity / 2`, for official tuple index `severity ∈ {0,1,2}` | `{0,0.5,1}` |
| defocus blur | `severity / 2`, for official tuple index `severity ∈ {0,1,2}` | `{0,0.5,1}` |
| JPEG artifact | `(30 - quality_factor) / (30 - 10)` | sampled values `{0.05,0.10,…,1.0}` because quality is integer `[10,30)` |
| low resolution | constant `1` for the fixed official x4 degradation | `{1}` |

Every scalar is clipped to `[0,1]` after evaluation. Motion/defocus use the min–max of the official ordered three-level severity control and retain the exact official tuple table in `reports/AGENTICIR_OPERATOR_PROTOCOL.md`. This is the frozen mapping used by the completed Stage1 lineage. It is not literal independent min–max normalization of defocus radius; that difference and its consequences are explicitly recorded in D-017 rather than silently rewritten after training.

## Absent skills

For each of the seven absent skills in a single-degradation episode, or each of the six absent skills in a Group-A pair episode:

```text
guard_target = 0 everywhere
global_severity_target = 0
presence_target = 0
```

Present skills receive `presence_target=1`. Duplicate skills in one recipe fail closed.

Stage2 effect-profile counterfactuals are the sole separate intervention: a deliberately forced absent skill uses a unit execution guard so that it is genuinely called while presence supervision remains absent. That policy is D-012 and does not alter the stored training guard target above.

## Geometry and alignment

1. Clean image, subset targets, and operator traces share the same deterministic recipe and per-operator seed.
2. Training crop is selected before degradation. Haze alone computes the official full-depth normalization and full operator output before taking the synchronized crop, preserving the locked full-image physics.
3. Input, clean/subset targets, and all eight guard planes receive the same horizontal flip, vertical flip, and 90-degree rotation.
4. Only after synchronized geometry, guards are reduced with FP32 `adaptive_avg_pool2d` to `(H//4, W//4)`.
5. Validation uses the complete image with no random augmentation.

## Fail-closed checks

- Operator source SHA, MiOIR source SHAs, manifest order, seeds, and parameter dictionaries are bound in run provenance.
- Missing haze transmission, missing before/after dense traces, duplicate skills, wrong shapes, non-BGR-uint8 operator images, or a source-hash mismatch raise an error.
- Guard targets are never loaded from MiO100, Group B, or Group C.
- The completed Stage1 run used only single and Group-A train manifests; formal validation used `primary_val` only.
