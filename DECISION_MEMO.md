# Decision Memo

Status: Stage3–4 and the one-shot formal Stage4 MiO100 A/B/C evaluation are complete. The selected step-40000 EMA checkpoint is frozen.

Stage1 finished at step30000 with Group-A `25.4706350267/0.7729285792`, single `28.9855093932/0.8669213048`, and pair-isolation `29.2419258004/0.8468425488`. Stage2 found a non-ambiguous parallel fraction of `0.6476190476`, serial-gap PSNR median/P75 of `0.1998348236/1.1096587181 dB`, and median pair-majority share `0.6079292929`. These observations support investigating sample-conditioned partial orders, but do not by themselves establish the final system's causal advantage.

The formal Stage4 MiO100 means are Group-A `24.3741960555/0.7599264849`, Group-B `19.8229607308/0.6678735027`, and Group-C `18.5247367167/0.5563379351` (PSNR/SSIM). Group A is strong relative to the published AgenticIR/RAR rows, while B/C expose weak unseen-composition generalization. On `primary_val`, Stage4 remains below Stage0 for both Single and Group-A, so V7.1 is engineering-complete but not a paper-ready proof of net system effectiveness. The next high-value experiment is a separately authorized, same-protocol formal Stage0 A/B/C evaluation; fair A1/A3/A4 controls remain unrun.

<!-- STAGE4_SSIM_RETENTION_BEGIN -->
## Stage4 SSIM retention

- SSIM_RETENTION_RISK: `true`
- Stage0 Group-A PSNR/SSIM: `24.809721372127534 / 0.785909488574689`
- Stage4 selected Group-A PSNR/SSIM: `24.546857466697695 / 0.7808854509814307`
- Delta PSNR/SSIM: `-0.26286390542983895 / -0.0050240375932583126`
- The selected Group-A PSNR does not offset the SSIM retention deficit.
- This disclosure did not alter checkpoint selection or the separately authorized one-shot MiO100 result.
<!-- STAGE4_SSIM_RETENTION_END -->
