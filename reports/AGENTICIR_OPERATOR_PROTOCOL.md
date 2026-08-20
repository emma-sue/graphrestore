# Locked AgenticIR Operator Protocol

- audit time (UTC): `2026-08-17T06:17:17Z`
- status: **PASS — locked sources match the V7.1 operator ranges**
- extraction method: static audit of the exact locked source, followed by SHA256 verification
- AgenticIR commit: `9640a291480dee3ba8f2974125d4ee9e3440f3d6`
- AgenticIR operator source: `/root/autodl-tmp/graph/upstream/AgenticIR/dataset/add_single_degradation.py`
- AgenticIR operator SHA256: `c97450a05acb805e59291a1335a743c77eca3db36f26a444b4033c7f6fe6369c`
- MiOIR commit: `4d5f6ca0235cf2c307319673242d5722ee35d73f`
- MiOIR degradation SHA256: `a507295ec9cbe47536bb7530f63ce385fb0ecb0c7b7fbe51b34b5db9d539d2fd`
- MiOIR `matlab_functions.py` SHA256: `29a3a3d209ce15724202bfb01415e5d4e574e7b853090551a7938c7b78ec4975`

## Extracted operator ranges

| Skill | Locked sampling and implementation |
|---|---|
| noise | Type is uniformly selected from `Gaussian` and `Poisson`. Gaussian passes `sigma_range=[20,50]`; Poisson passes `scale_range=[1,3]` to the locked BasicSR functions, with clipping enabled and internal rounding disabled before the final BGR uint8 conversion. |
| jpeg artifact | `quality_factor = np.random.randint(10,30)`, hence integer quality in `[10,30)`, followed by OpenCV JPEG encode/decode in BGR. |
| low light (`dark`) | Type is selected from constant shift, gamma correction, and linear mapping. Shift is integer `[30,50)`; gamma is continuous uniform `[0.5,0.7)`; linear `dst_max` is integer `[100,150)`. The operation is applied to the HSV value channel and returns BGR uint8. |
| haze | `A ~ U(0.7,1.0)` and `beta ~ U(0.6,1.8)`. MiOIR depth is resized by OpenCV `INTER_CUBIC` at `fx=fy=4`, normalized by its full-image maximum, and used in `t=exp(-beta*d)` and `I=J*t+A*255*(1-t)`. |
| motion blur | `severity ∈ {0,1,2}` selects `(radius,sigma)` from `[(10,3),(15,5),(15,8)]`. Kernel width is `2*radius+1`; angle is continuous uniform `[-90,90)`. |
| defocus blur | `severity ∈ {0,1,2}` selects `(radius,alias_blur)` from `[(3,0.1),(4,0.5),(6,0.5)]`. |
| rain | Length is integer `[20,40)`, angle is integer `[-30,30)`, and value is integer `[50,100)`. Width is fixed at three. The visible clipped BGR uint8 result is authoritative. |
| low resolution | Locked BasicSR/MiOIR `imresize(scale=0.25)` produces native one-quarter-resolution BGR uint8. GraphRestore then applies the separately frozen online canonical x4 float path; the operator itself is never replaced with OpenCV resize. |

All interval endpoints above retain the semantics of the exact NumPy or BasicSR call in the locked source. No range was inferred from observed recipes, widened, clipped to a new interval, or replaced by an empirical setting.

## Replay rules

- Every operator runs in the recorded manifest order with its own uint32 seed.
- The RNG transaction restores Python, NumPy, Torch CPU, and worker-generator state after each call.
- Rain and motion blur are replayed through the original random call sequence rather than passing an argument that would skip an upstream draw and change later random values.
- Haze reuses the recorded `A` and `beta` and the read-only `<clean_id>/predict_depth.mat` compatibility link.
- All degradation composition remains in BGR uint8. RGB float conversion happens only at the model boundary.
- The isolated reference parity audit produced byte-exact BGR uint8 output for two real primary-train recipes for each of the eight operators (`16/16` pairs); see `reports/DEGRADATION_PROTOCOL.md`.

## Guard-normalization boundary

This report freezes the official **operator** parameter ranges. Guard-target normalization is a separate training protocol and is recorded in `reports/GUARD_PROTOCOL.md`. In particular, the already-trained lineage represents the three official motion/defocus tuples by their ordinal severity level (`severity/2`). The distinction from literal parameter-wise radius normalization is disclosed as D-017 in `reports/DEVIATIONS.md`; no operator pixels or official parameter ranges are changed.
