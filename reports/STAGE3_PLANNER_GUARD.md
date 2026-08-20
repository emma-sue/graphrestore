# Stage3 Planner and Guard

- protocol: `graphrestore-v7.1-agenticir-locked`
- finalization mode: `step12000_finalize_only_no_training`
- selected step: 12000
- selected best checkpoint: `/root/autodl-tmp/aaa/graphrestore/artifacts/checkpoints/stage3/best_ema.pth`
- selected best checkpoint SHA256: `9114974f68f202119d4241077d0c46333315204959d58b7eabecaf68a3e32ff3`
- Stage4 parent role: only_stage3_parent
- finalization authorization SHA256: `a4d3abb112aae01afeef5c87f3ffb65a26f3ff8cdff1cc8ad362db2323d9743e`
- frozen thresholds: `/root/autodl-tmp/aaa/graphrestore/artifacts/planner_thresholds.json`
- optimizer / scheduler / train loader created: false / false / false
- checkpoint written: false
- optimizer steps executed / checkpoint writes / sampler steps advanced: 0 / 0 / 0
- step14000 pending checkpoint role: abandoned_unselected_extension_state
- MiO100 / Group B / Group C rows read: 0 / 0 / 0
- original selection and six-validation history remain byte-exact

## Original selected diagnostic at threshold 0.50 (selection unchanged)

- Single PSNR/SSIM: 25.4050959134 / 0.8309895028
- Group-A PSNR/SSIM: 22.0272254491 / 0.7118815277

## Post-calibration full primary_val diagnostic

- Single PSNR/SSIM: 25.5378938150 / 0.8390662570
- Group-A PSNR/SSIM: 22.6091022968 / 0.7331561477
- planner macro F1: 0.9086336704
- macro F1 before/after calibration: 0.8454899764 / 0.9086336704
- planner activation rate (skill slots): 0.2015624940
- noise threshold/P/R/F1/activation: 0.42 / 0.9829351536 / 0.9600000000 / 0.9713322091 / 0.1831250042
- motion_blur threshold/P/R/F1/activation: 0.48 / 0.5851528384 / 0.8933333333 / 0.7071240106 / 0.2862499952
- defocus_blur threshold/P/R/F1/activation: 0.50 / 0.9081632653 / 0.8900000000 / 0.8989898990 / 0.1837500036
- jpeg_artifact threshold/P/R/F1/activation: 0.50 / 1.0000000000 / 1.0000000000 / 1.0000000000 / 0.1875000000
- rain threshold/P/R/F1/activation: 0.38 / 0.9603960396 / 0.9700000000 / 0.9651741294 / 0.1893749982
- haze threshold/P/R/F1/activation: 0.44 / 0.8836477987 / 0.9366666667 / 0.9093851133 / 0.1987500042
- low_light threshold/P/R/F1/activation: 0.48 / 0.8115015974 / 0.8466666667 / 0.8287112561 / 0.1956250072
- low_resolution threshold/P/R/F1/activation: 0.42 / 0.9867109635 / 0.9900000000 / 0.9883527454 / 0.1881249994
- learned raw relation accuracy: 0.6231292517
- learned raw relation macro-F1/balanced accuracy: 0.5090908920 / 0.5132586311
- parallel precision/recall: 0.7425531915 / 0.7331932773
- always-parallel baseline accuracy: 0.6476190476
- always-parallel baseline macro-F1/balanced accuracy: 0.2620423892 / 0.3333333333
- per-pair majority-prior baseline accuracy: 0.6612244898
- per-pair majority-prior macro-F1/balanced accuracy: 0.4284525416 / 0.4653793024
- rain guard Spearman/MAE: 0.5344731333 / 0.0804204962
- haze guard Spearman/MAE: 0.5503607199 / 0.1132148327
- mean program levels: 1.2400000000
- STOP rate: 0.0125000000
- STOP-rate definition: fraction of primary_val samples whose stopped_mask fired in any formal inference round
- post-compiler cycle rate: 0.0000000000
