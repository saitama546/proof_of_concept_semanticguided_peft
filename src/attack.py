# src/attack.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.losses import gradient_matching_loss, total_variation_loss
from src.clip_prior import CLIPTextPrior


def _reconstruct_with_assumed_label(
    model: torch.nn.Module,
    observed_grads: List[torch.Tensor],
    assumed_label: int,
    image_shape: tuple,
    device: torch.device,
    mode: str,
    prompt: Optional[str],
    clip_prior: Optional[CLIPTextPrior],
    steps: int,
    lr: float,
    tv_weight: float,
    semantic_weight: float,
    log_every: int,
) -> Dict:
    model.eval()
    adapter_params = model.get_adapter_parameters()

    dummy_logits = torch.randn(
        image_shape,
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([dummy_logits], lr=lr)

    label_tensor = torch.tensor(
        [assumed_label],
        dtype=torch.long,
        device=device,
    )

    text_features = None
    if mode in {"correct_text", "wrong_text"}:
        if prompt is None or clip_prior is None:
            raise ValueError(f"Mode {mode} requires prompt and clip_prior.")
        text_features = clip_prior.encode_text(prompt)

    history = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        dummy_x = torch.sigmoid(dummy_logits)

        logits = model(dummy_x)
        loss_cls = F.cross_entropy(logits, label_tensor)

        dummy_grads = torch.autograd.grad(
            loss_cls,
            adapter_params,
            create_graph=True,
            retain_graph=True,
        )

        grad_loss = gradient_matching_loss(
            list(dummy_grads),
            observed_grads,
        )

        semantic_loss = torch.tensor(0.0, device=device)

        if mode in {"correct_text", "wrong_text"}:
            semantic_loss = clip_prior.text_image_loss(
                image=dummy_x,
                text_features=text_features,
            )

        tv_loss = total_variation_loss(dummy_x)

        total_loss = (
            grad_loss
            + semantic_weight * semantic_loss
            + tv_weight * tv_loss
        )

        total_loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(
                f"label={assumed_label} | "
                f"step={step:04d} | "
                f"mode={mode} | "
                f"total={total_loss.item():.6f} | "
                f"grad={grad_loss.item():.6f} | "
                f"semantic={semantic_loss.item():.6f} | "
                f"tv={tv_loss.item():.6f}"
            )

        history.append(
            {
                "step": step,
                "assumed_label": assumed_label,
                "total_loss": float(total_loss.detach().cpu()),
                "grad_loss": float(grad_loss.detach().cpu()),
                "semantic_loss": float(semantic_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
            }
        )

    reconstructed = torch.sigmoid(dummy_logits).detach()
    final_grad_loss = history[-1]["grad_loss"]

    return {
        "reconstruction": reconstructed,
        "history": history,
        "assumed_label": assumed_label,
        "final_gradient_loss": final_grad_loss,
    }


def _reconstruct_text_only(
    image_shape: tuple,
    device: torch.device,
    prompt: str,
    clip_prior: CLIPTextPrior,
    steps: int,
    lr: float,
    tv_weight: float,
    semantic_weight: float,
    log_every: int,
) -> Dict:
    dummy_logits = torch.randn(
        image_shape,
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([dummy_logits], lr=lr)
    text_features = clip_prior.encode_text(prompt)

    history = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        dummy_x = torch.sigmoid(dummy_logits)

        grad_loss = torch.tensor(0.0, device=device)

        semantic_loss = clip_prior.text_image_loss(
            image=dummy_x,
            text_features=text_features,
        )

        tv_loss = total_variation_loss(dummy_x)

        total_loss = (
            semantic_weight * semantic_loss
            + tv_weight * tv_loss
        )

        total_loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(
                f"step={step:04d} | "
                f"mode=text_only | "
                f"total={total_loss.item():.6f} | "
                f"grad={grad_loss.item():.6f} | "
                f"semantic={semantic_loss.item():.6f} | "
                f"tv={tv_loss.item():.6f}"
            )

        history.append(
            {
                "step": step,
                "assumed_label": None,
                "total_loss": float(total_loss.detach().cpu()),
                "grad_loss": float(grad_loss.detach().cpu()),
                "semantic_loss": float(semantic_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
            }
        )

    reconstructed = torch.sigmoid(dummy_logits).detach()

    return {
        "reconstruction": reconstructed,
        "history": history,
        "assumed_label": None,
        "final_gradient_loss": None,
    }


def reconstruct_from_adapter_gradient(
    model: torch.nn.Module,
    observed_grads: Optional[List[torch.Tensor]],
    image_shape: tuple,
    device: torch.device,
    mode: str = "grad_only",
    prompt: Optional[str] = None,
    clip_prior: Optional[CLIPTextPrior] = None,
    candidate_labels: Optional[Sequence[int]] = None,
    steps: int = 500,
    lr: float = 0.05,
    tv_weight: float = 1e-4,
    semantic_weight: float = 1e-3,
    log_every: int = 50,
) -> Dict:
    """
    Reconstruct private image without giving the attacker the true label.

    For gradient-based modes, attacker searches candidate labels and selects
    the one with lowest final gradient mismatch.

    Modes:
        grad_only:
            gradient matching only

        correct_text:
            gradient matching + correct text prompt

        wrong_text:
            gradient matching + wrong text prompt

        text_only:
            CLIP text prompt only; no gradient, no label search
    """
    valid_modes = {"grad_only", "correct_text", "wrong_text", "text_only"}

    if mode not in valid_modes:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")

    if candidate_labels is None:
        candidate_labels = list(range(10))

    if mode in {"grad_only", "correct_text", "wrong_text"}:
        if observed_grads is None:
            raise ValueError(f"Mode {mode} requires observed adapter gradients.")

    if mode in {"correct_text", "wrong_text", "text_only"}:
        if prompt is None:
            raise ValueError(f"Mode {mode} requires a text prompt.")
        if clip_prior is None:
            raise ValueError(f"Mode {mode} requires a CLIPTextPrior object.")

    if mode == "text_only":
        return _reconstruct_text_only(
            image_shape=image_shape,
            device=device,
            prompt=prompt,
            clip_prior=clip_prior,
            steps=steps,
            lr=lr,
            tv_weight=tv_weight,
            semantic_weight=semantic_weight,
            log_every=log_every,
        )

    best_result = None
    all_label_results = []

    for label in candidate_labels:
        print("\n" + "=" * 80)
        print(f"Trying assumed label: {label}")
        print("=" * 80)

        result = _reconstruct_with_assumed_label(
            model=model,
            observed_grads=observed_grads,
            assumed_label=int(label),
            image_shape=image_shape,
            device=device,
            mode=mode,
            prompt=prompt,
            clip_prior=clip_prior,
            steps=steps,
            lr=lr,
            tv_weight=tv_weight,
            semantic_weight=semantic_weight,
            log_every=log_every,
        )

        all_label_results.append(
            {
                "assumed_label": int(label),
                "final_gradient_loss": result["final_gradient_loss"],
            }
        )

        if best_result is None:
            best_result = result
        elif result["final_gradient_loss"] < best_result["final_gradient_loss"]:
            best_result = result

    best_result["all_label_results"] = all_label_results

    print("\nLabel search summary:")
    for item in all_label_results:
        print(
            f"label={item['assumed_label']} | "
            f"final_gradient_loss={item['final_gradient_loss']:.8f}"
        )

    print(f"\nSelected label: {best_result['assumed_label']}")

    return best_result