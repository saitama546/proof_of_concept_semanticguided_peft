# src/metrics.py

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Description:
        Convert a torch.Tensor image of shape [1, 3, H, W] in [0, 1] to a numpy array of shape [H, W, 3] in [0, 1]. datatype :np.float32.
    INPUT:
        x: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    OUTPUT:
        np.ndarray of shape [H, W, 3] in [0, 1]. datatype :np.float32.
    """
    x = x.detach().cpu().squeeze(0)
    x = x.permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)


def compute_psnr_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict:
    """
    Description:
        Compute PSNR and SSIM between original and reconstructed images.
    INPUT:
    - original: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    - reconstructed: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    OUTPUT:
    - dict with keys "psnr" and "ssim".
    """
    original_np = tensor_to_numpy_image(original)
    reconstructed_np = tensor_to_numpy_image(reconstructed)

    psnr = peak_signal_noise_ratio(
        original_np,
        reconstructed_np,
        data_range=1.0,
    )

    ssim = structural_similarity(
        original_np,
        reconstructed_np,
        channel_axis=2,
        data_range=1.0,
    )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


def compute_lpips(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    device: torch.device,
    net: str = "alex",
) -> float:
    """
    Description:
        Compute LPIPS between original and reconstructed images.
    INPUT: 
    - original: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    - reconstructed: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    - device: torch.device to run LPIPS on.
    - net: str, one of "alex", "vgg", or "squeeze". The backbone network to use for LPIPS.
    OUTPUT:
    - float, the LPIPS value between original and reconstructed images.
    """
    lpips_model = lpips.LPIPS(net=net).to(device)
    lpips_model.eval()

    with torch.no_grad():
        original_lpips = original * 2.0 - 1.0
        reconstructed_lpips = reconstructed * 2.0 - 1.0

        value = lpips_model(original_lpips, reconstructed_lpips)

    return float(value.item())


def compute_all_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Description:
        Compute PSNR, SSIM, and LPIPS between original and reconstructed images.
    INPUT:
    - original: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    - reconstructed: torch.Tensor of shape [1, 3, H, W] in [0, 1]. datatype :torch.float32.
    - device: torch.device to run LPIPS on.
    OUTPUT:
    - dict with keys "psnr", "ssim", and "lpips".
    """
    result = compute_psnr_ssim(
        original=original,
        reconstructed=reconstructed,
    )

    result["lpips"] = compute_lpips(
        original=original,
        reconstructed=reconstructed,
        device=device,
    )

    return result