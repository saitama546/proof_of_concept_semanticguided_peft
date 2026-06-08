# scripts/run_gradient_reconstruction.py

import json
import sys
from pathlib import Path

from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import load_cifar10, get_private_sample
from src.model import ViTAdapterClassifier
from src.fl import FLServer, FLClient
from src.train import calibrate_adapter_classifier
from src.attack import reconstruct_from_adapter_gradient
from src.metrics import compute_all_metrics
from src.clip_prior import CLIPTextPrior
from src.utils import set_seed, get_device


def main():
    set_seed(0)
    device = get_device()

    print(f"Using device: {device}")

    image_dir = ROOT / "results" / "images"
    metric_dir = ROOT / "results" / "metrics"
    image_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Choose attack mode here
    # Options:
    #   "grad_only"
    #   "correct_text"
    #   "wrong_text"
    #   "text_only"
    # ------------------------------------------------------------
    mode = "text_only"

    valid_modes = {"grad_only", "correct_text", "wrong_text", "text_only"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")

    # ------------------------------------------------------------
    # Experiment settings
    # ------------------------------------------------------------
    target_class = "frog"

    correct_prompt = "a photo of a frog"
    wrong_prompt = "a photo of a truck"

    # For debugging after label-search change, start with 100.
    # Later increase to 500.
    steps = 100

    lr = 0.05
    tv_weight = 1e-4
    semantic_weight = 1e-3

    # CIFAR-10 label candidates.
    # Attacker knows the task label space, not the true private label.
    candidate_labels = list(range(10))

    # ------------------------------------------------------------
    # 1. Load CIFAR-10
    # ------------------------------------------------------------
    train_dataset = load_cifar10(
        root=str(ROOT / "data"),
        train=True,
        image_size=224,
        download=True,
    )

    test_dataset = load_cifar10(
        root=str(ROOT / "data"),
        train=False,
        image_size=224,
        download=True,
    )

    # ------------------------------------------------------------
    # 2. Select one held-out private target image from the test split
    # ------------------------------------------------------------
    private_sample = get_private_sample(
        dataset=test_dataset,
        class_name=target_class,
        device=device,
    )

    original_path = image_dir / f"{target_class}_original.png"
    save_image(
        private_sample.image.detach().cpu(),
        original_path,
    )

    print("\nPrivate sample:")
    print(f"Target class: {private_sample.class_name}")
    print(f"Dataset index: {private_sample.dataset_index}")
    print(f"True label for evaluation only: {private_sample.label.item()}")
    print(f"Image shape: {tuple(private_sample.image.shape)}")
    print(f"Saved original image to: {original_path}")

    # ------------------------------------------------------------
    # 3. Server builds global model:
    #    pretrained frozen ViT backbone + trainable adapter + classifier head
    # ------------------------------------------------------------
    global_model = ViTAdapterClassifier(
        backbone_name="vit_tiny_patch16_224",
        num_classes=10,
        bottleneck_dim=32,
        pretrained=True,
    ).to(device)

    # ------------------------------------------------------------
    # 4. Calibrate adapter + classifier on public/train split.
    #    Private target image is from test split, so it is not used here.
    # ------------------------------------------------------------
    print("\nCalibrating adapter + classifier...")

    global_model = calibrate_adapter_classifier(
        model=global_model,
        train_dataset=train_dataset,
        device=device,
        num_samples=1000,
        batch_size=32,
        epochs=2,
        lr=1e-3,
    )

    # ------------------------------------------------------------
    # 5. Server sends calibrated global model copy to Client 1
    # ------------------------------------------------------------
    server = FLServer(global_model=global_model)

    client_1 = FLClient(
        client_id=1,
        model=server.send_model_to_client(),
        device=device,
    )

    # ------------------------------------------------------------
    # 6. Client 1 computes local adapter gradient from private image.
    #
    # This is client-side. Client is allowed to use its own true label.
    # The attacker will NOT receive this label.
    # ------------------------------------------------------------
    observed_grads = client_1.compute_adapter_gradient(
        private_x=private_sample.image,
        private_y=private_sample.label,
    )

    print("\nObserved Client 1 adapter gradient norms:")
    for i, grad in enumerate(observed_grads):
        print(f"grad {i}: shape={tuple(grad.shape)}, norm={grad.norm().item():.6f}")

    # ------------------------------------------------------------
    # 7. Prepare CLIP semantic prior if needed
    # ------------------------------------------------------------
    clip_prior = None
    prompt = None

    if mode in {"correct_text", "wrong_text", "text_only"}:
        print("\nLoading CLIP semantic prior...")
        clip_prior = CLIPTextPrior(device=device)

    if mode == "correct_text":
        prompt = correct_prompt
    elif mode == "wrong_text":
        prompt = wrong_prompt
    elif mode == "text_only":
        prompt = correct_prompt

    print("\nAttack setting:")
    print(f"Mode: {mode}")
    print(f"Prompt: {prompt}")
    print(f"Candidate labels: {candidate_labels}")
    print(f"Steps per candidate label: {steps}")
    print(f"LR: {lr}")
    print(f"TV weight: {tv_weight}")
    print(f"Semantic weight: {semantic_weight}")

    # ------------------------------------------------------------
    # 8. Attacker reconstructs image.
    #
    # Important:
    # private_sample.label is NOT passed to the attacker.
    # The attacker searches over candidate_labels instead.
    # ------------------------------------------------------------
    print(f"\nStarting reconstruction with mode={mode}...")

    result = reconstruct_from_adapter_gradient(
        model=client_1.model,
        observed_grads=observed_grads if mode != "text_only" else None,
        image_shape=tuple(private_sample.image.shape),
        device=device,
        mode=mode,
        prompt=prompt,
        clip_prior=clip_prior,
        candidate_labels=candidate_labels,
        steps=steps,
        lr=lr,
        tv_weight=tv_weight,
        semantic_weight=semantic_weight,
        log_every=50,
    )

    reconstruction = result["reconstruction"]

    recon_path = image_dir / f"{target_class}_{mode}_reconstruction.png"
    save_image(
        reconstruction.detach().cpu(),
        recon_path,
    )

    print(f"\nSaved reconstruction to: {recon_path}")

    # ------------------------------------------------------------
    # 9. Compute metrics: PSNR, SSIM, LPIPS
    #
    # Original private image is used only for evaluation.
    # ------------------------------------------------------------
    metrics = compute_all_metrics(
        original=private_sample.image,
        reconstructed=reconstruction,
        device=device,
    )

    metrics["target_class"] = target_class
    metrics["mode"] = mode
    metrics["prompt"] = prompt
    metrics["steps"] = steps
    metrics["lr"] = lr
    metrics["tv_weight"] = tv_weight
    metrics["semantic_weight"] = semantic_weight

    # Label search information
    metrics["selected_label"] = result.get("assumed_label")
    metrics["true_label_for_evaluation_only"] = int(private_sample.label.item())
    metrics["all_label_results"] = result.get("all_label_results")

    # Add final reconstruction losses from attack history
    if "history" in result and len(result["history"]) > 0:
        final_record = result["history"][-1]
        metrics["final_total_loss"] = final_record["total_loss"]
        metrics["final_gradient_loss"] = final_record["grad_loss"]
        metrics["final_semantic_loss"] = final_record["semantic_loss"]
        metrics["final_tv_loss"] = final_record["tv_loss"]

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    metrics_path = metric_dir / f"{target_class}_{mode}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")

    print("\nRun completed.")
    print(f"Finished mode: {mode}")


if __name__ == "__main__":
    main()