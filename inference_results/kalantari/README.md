# Kalantari/SIG17 inference results

Predictions from the principal EMCA-HDR checkpoint (`ckpt_EMCA/PSNR_mu44.5961.pth`) for the 15 scenes in the SIG17/Kalantari test split. The dataset itself is not included.

For each scene, `*_pred.hdr` is the linear HDR prediction and `*_pred.png` is its tone-mapped visualization.

## Mean metrics across 15 scenes

| Metric | Mean |
|---|---:|
| PSNR (linear) | 42.0628 dB |
| SSIM (linear) | 0.9891 |
| PSNR (mu-law) | 44.5986 dB |
| SSIM (mu-law) | 0.9917 |

SSIM values use the implementation's expected `[0, 255]` input scale. Metrics were computed against `HDRImg.hdr` references; PSNR values are from the in-memory predictions, while SSIM was recalculated from the saved HDR predictions.

Inference was run on CPU with PyTorch 2.5.1. Model forward time averaged 22.23 seconds per scene (about 7 minutes 22 seconds total, including data loading and metrics). The original dataset files are not redistributed here.
