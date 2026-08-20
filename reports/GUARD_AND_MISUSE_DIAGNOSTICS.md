# Stage4 Guard and Misuse Diagnostics

- Selected EMA SHA256: `6aa2de6e65ce633430d188857845acef1b67cfb3b218a04977293dbc149a84fd`
- Optimizer updates: `0`
- Model/EMA/RNG unchanged: `true`
- Dataset: frozen primary_val single + Group A only

| Family | Mode | Group-A PSNR / SSIM | Single PSNR / SSIM |
|---|---|---:|---:|
| compiler_modes | full_partial_order | 24.546857 / 0.78088545 | 27.020805 / 0.86857653 |
| compiler_modes | forced_total_order | 24.462585 / 0.77450225 | 27.076748 / 0.86800731 |
| compiler_modes | parallel_only | 24.284616 / 0.77727398 | 26.842399 / 0.86724450 |
| guard_modes | predicted_spatial | 24.546857 / 0.78088545 | 27.020805 / 0.86857653 |
| guard_modes | global_mean | 24.522510 / 0.77904532 | 26.925415 / 0.86737470 |
| guard_modes | all_one | 21.513926 / 0.69413600 | 23.001584 / 0.76417021 |
