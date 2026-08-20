# Stage1 Guarded Skill Bank

- Protocol: `graphrestore-v7.1-agenticir-locked`
- Validation step: 30000
- Validation data: frozen `primary_val` singles + Group A only (no MiO100)
- Metric: AgenticIR parity PSNR-RGB / SSIM-Y, output clamp-round-uint8
- Selected EMA checkpoint: `/root/autodl-tmp/aaa/graphrestore/artifacts/checkpoints/stage1/best_ema.pth`
- Best Group-A PSNR/SSIM: 25.470635 / 0.77292858

## Episode metrics

| Episode | Count | PSNR | SSIM | Residual norm | Active rate |
|---|---:|---:|---:|---:|---:|
| single_skill | 800 | 28.985509 | 0.86692130 | 0.07115191 | 0.125000 |
| pair_isolation | 1600 | 29.241926 | 0.84684255 | 0.06343053 | 0.125000 |
| pair_parallel | 800 | 25.470635 | 0.77292858 | 0.10325485 | 0.250000 |

## Per-skill activation diagnostics

| Skill | Evaluations | PSNR | SSIM | Residual norm | Active rate |
|---|---:|---:|---:|---:|---:|
| noise | 500 | 29.005862 | 0.83131223 | 0.09381381 | 0.175000 |
| motion_blur | 500 | 23.516379 | 0.72961031 | 0.03405433 | 0.175000 |
| defocus_blur | 500 | 26.683063 | 0.78783219 | 0.05172164 | 0.175000 |
| jpeg_artifact | 500 | 28.265491 | 0.80629854 | 0.02690828 | 0.175000 |
| rain | 500 | 31.188605 | 0.88067769 | 0.09090038 | 0.175000 |
| haze | 500 | 28.962134 | 0.93360799 | 0.15330666 | 0.175000 |
| low_light | 500 | 26.758329 | 0.83060084 | 0.16303087 | 0.175000 |
| low_resolution | 500 | 27.077147 | 0.77040191 | 0.03350030 | 0.175000 |
