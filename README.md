# EMCA-HDR

Official implementation and pretrained weights for **EMCA-HDR**. This repository includes the principal model checkpoint and checkpoints from the ablation study. The SIG17 dataset is not included; obtain it from its official source and arrange it as described below.

## Repository contents

- `models/ferreti.py`: EMCA-HDR model implementation.
- `ckpt_EMCA/PSNR_mu44.5961.pth`: principal pretrained checkpoint.
- `Ablation_Study/`: ablation model definitions, checkpoints, and validation results.
- `test.py`: inference and evaluation script.
- `dataset/` and `utils/`: data loading and image/metric utilities.

The VGG16 weights are not bundled. `torchvision` downloads them automatically if they are needed by the training loss.

The optional HDR-VDP MATLAB toolkit is not bundled because its files have separate redistribution terms; obtain it from its upstream project if needed.

## Installation

Python 3.9 or newer is recommended. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version from the [official PyTorch instructions](https://pytorch.org/get-started/locally/). CPU inference is also supported.

## Dataset layout

Place the SIG17 test set under `data/Test/` (or supply another root with `--data_root`). Each scene directory must contain three exposure-ordered `.tif` images and an `exposure.txt` file with one log-exposure value per line. For metric evaluation, include the reference image as `HDRImg.hdr`:

```text
data/
└── Test/
    └── SceneName/
        ├── 0.tif
        ├── 1.tif
        ├── 2.tif
        ├── exposure.txt
        └── HDRImg.hdr
```

The `.tif` images should be named so alphabetical order matches exposure order. `HDRImg.hdr` is used as ground truth for metrics. If the reference is absent, the dataset loader falls back to the middle-exposure image; those metrics should not be treated as reference evaluation.

## Inference

From the repository root, run inference with the included principal checkpoint:

```bash
python test.py
```

The script uses `data/`, `ckpt_EMCA/PSNR_mu44.5961.pth`, and `test_results/inference/` by default. Override these paths as needed:

```bash
python test.py \
  --data_root /path/to/dataset \
  --pretrained_model ckpt_EMCA/PSNR_mu44.5961.pth \
  --save_dir /path/to/predictions \
  --num_workers 4
```

To force CPU inference, add `--no_cuda`. The script saves one tone-mapped PNG and one linear HDR prediction per scene, and reports PSNR/SSIM in linear and mu-law domains when reference HDR images are available. Optional model complexity reporting is available with `--profile` after installing `thop` (`pip install thop`).

## Ablation checkpoints

The model variants and their weights are in `Ablation_Study/ckpt_Ferreti(...)`. Each folder's `best_checkpoint_.pth` is the best checkpoint for that variant; `latest_checkpoint_.pth` is the final checkpoint. The corresponding `xxxx.py` file defines that variant. These checkpoints are provided for research comparison; the standard inference script loads the principal EMCA-HDR model.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE). Please also observe the terms of the SIG17 dataset and any third-party code/assets used in your work.
