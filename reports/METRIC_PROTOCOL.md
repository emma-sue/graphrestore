# AgenticIR Official Metric Protocol and Parity

- protocol: `graphrestore-v7.1-agenticir-metric-parity`
- created_utc: `2026-08-14T21:20:26Z`
- result: **PASS**
- failures: `0`
- warnings: `0`

## Checks

- **PASS** `metric.psnr.max_abs` — max_abs=0
- **PASS** `metric.ssim.max_abs` — max_abs=3.87908013377e-07
- **PASS** `low_resolution.canonical_float_exact` — 8 pairs byte-identical
- **PASS** `low_resolution.canonical_uint8_exact` — 8 pairs byte-identical

## Machine-readable facts

```json
{
  "canonical_float_exact": true,
  "canonical_uint8_exact": true,
  "full_size_pairs": 16,
  "max_psnr_abs_diff": 0.0,
  "max_ssim_abs_diff": 3.8790801337729164e-07,
  "native_x4_pairs": 8,
  "versions": {
    "agenticir_commit": "9640a291480dee3ba8f2974125d4ee9e3440f3d6",
    "agenticir_scorer_sha256": "b6eee989575ee17d2cbf9e38fbab0a996b54a5260ae205246c718c08facab830",
    "opencv_fast": "5.0.0",
    "reference_environment": {
      "basicsr": "1.4.2",
      "numpy": "1.26.4",
      "opencv": "4.9.0",
      "pyiqa": "0.1.10",
      "python": "3.12.3",
      "torch": "2.3.0+cu121"
    },
    "reference_python": "/root/autodl-tmp/aaa/graphrestore/.venv-reference/bin/python",
    "torch_fast": "2.3.0+cu121"
  }
}
```
