#Purpose:   
# src/losses.py

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def gradient_matching_loss(
    dummy_grads: List[torch.Tensor],
    observed_grads: List[torch.Tensor],
) -> torch.Tensor:
    """
    Description:Compute the MSE loss between dummy gradients and observed gradients.
    INPUTS:
    - dummy_grads: List of tensors representing the dummy gradients (e.g., from a reconstructed image). datatype: List[torch.Tensor]
    - observed_grads: List of tensors representing the observed gradients (e.g., from the real private image). datatype: List[torch.Tensor]
    OUTPUT:
    - A single scalar tensor representing the total gradient matching loss. datatype: torch.Tensor
    """
    
    loss = torch.tensor(0.0, device=dummy_grads[0].device)

    for dummy_grad, observed_grad in zip(dummy_grads, observed_grads):
        loss = loss + F.mse_loss(dummy_grad, observed_grad)

    return loss


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Desçription:Compute the total variation loss for an image tensor. This loss encourages spatial smoothness in the reconstructed image.
    INPUTS:
    - x: A tensor of shape (1, 3, H, W) representing the reconstructed image. datatype: torch.Tensor
    OUTPUT:
    - A single scalar tensor representing the total variation loss. datatype: torch.Tensor  
    """
    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return diff_h + diff_w