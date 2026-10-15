"""Run EMCA-HDR inference on the SIG17 test split."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_sig17 import SIG17_Validation_Dataset
from models.ferreti import ferreti
from utils.utils import AverageMeter, calculate_ssim, psnr, range_compressor


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run EMCA-HDR inference")
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data",
                        help="Dataset root containing the Test/ directory")
    parser.add_argument("--pretrained_model", type=Path,
                        default=PROJECT_ROOT / "ckpt_EMCA" / "PSNR_mu44.5961.pth",
                        help="Path to the pretrained EMCA-HDR checkpoint")
    parser.add_argument("--save_dir", type=Path,
                        default=PROJECT_ROOT / "test_results" / "inference",
                        help="Directory where predictions will be saved")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU inference")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data-loading worker processes")
    parser.add_argument("--profile", action="store_true",
                        help="Print FLOPs and parameter count (requires thop)")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    state_dict = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model


def main():
    args = parse_args()
    test_dir = args.data_root / "Test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test split not found at {test_dir}. Expected --data_root to contain Test/.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}")

    model = ferreti().to(device)
    model = load_checkpoint(model, args.pretrained_model, device)
    model.eval()

    dataset = SIG17_Validation_Dataset(args.data_root, crop=False)
    if len(dataset) == 0:
        raise RuntimeError(f"No test scenes found in {test_dir}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    psnr_linear, ssim_linear = AverageMeter(), AverageMeter()
    psnr_mu, ssim_mu = AverageMeter(), AverageMeter()
    elapsed_times = []

    for index, sample in enumerate(tqdm(loader, desc="Inference")):
        inputs = [sample[f"input{i}"].to(device) for i in range(3)]
        label = sample["label"].to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(*inputs)
        elapsed_times.append(time.perf_counter() - start)

        pred = prediction[0].cpu().permute(1, 2, 0).numpy()
        target = label[0].cpu().permute(1, 2, 0).numpy()
        pred_mu, target_mu = range_compressor(pred), range_compressor(target)

        psnr_linear.update(psnr(pred, target))
        ssim_linear.update(calculate_ssim(pred, target))
        psnr_mu.update(psnr(pred_mu, target_mu))
        ssim_mu.update(calculate_ssim(pred_mu, target_mu))

        scene_name = dataset.scenes_list[index]
        cv2.imwrite(str(args.save_dir / f"{scene_name}_pred.png"),
                    np.clip(pred_mu[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(str(args.save_dir / f"{scene_name}_pred.hdr"), pred[:, :, ::-1].astype(np.float32))

    print(f"PSNR (linear): {psnr_linear.avg:.4f} dB | SSIM (linear): {ssim_linear.avg:.4f}")
    print(f"PSNR (mu-law): {psnr_mu.avg:.4f} dB | SSIM (mu-law): {ssim_mu.avg:.4f}")
    print(f"Average inference time: {np.mean(elapsed_times):.4f}s/image")
    print(f"Predictions saved to: {args.save_dir}")

    if args.profile:
        try:
            from thop import profile
        except ImportError as error:
            raise RuntimeError("Install thop to use --profile: pip install thop") from error
        sample_inputs = tuple(torch.randn(1, 6, 128, 128, device=device) for _ in range(3))
        flops, params = profile(model, inputs=sample_inputs, verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f} G | Parameters: {params / 1e6:.2f} M")


if __name__ == "__main__":
    main()
