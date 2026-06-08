# scripts/test_calibrated_gradient.py

import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import load_cifar10, get_private_sample, split_dataset_two_clients
from src.model import ViTAdapterClassifier
from src.fl import FLServer, FLClient
from src.train import calibrate_adapter_classifier, get_trainable_parameter_status
from src.utils import set_seed, get_device


def main():
    """
    Description:Purpose of this script is to test whether a real private image from CIFAR-10 can produce adapter
    gradients on the client side in a federated learning setup. The script loads CIFAR-10, creates two simulated
    clients, selects a held-out private image, and checks that the client can compute adapter gradients from that image.
    The script includes assertions to verify that adapter gradients are produced and prints their shapes and norms for verification.
    INPUTS: None (the script uses CIFAR-10 dataset which is downloaded if not already present)
    OUTPUT:     - Prints the status of the server and clients, including model parameter statuses and adapter gradients.
                - Assertions to verify that adapter gradients are produced from the real CIFAR-10 private image.    
    """
    set_seed(0)
    device = get_device()

    print(f"Using device: {device}")

    image_dir = ROOT / "results" / "images"
    checkpoint_dir = ROOT / "results" / "checkpoints"
    image_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CIFAR-10
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

    # 2. Select held-out private target image from test split
    target_class = "frog"

    private_sample = get_private_sample(
        dataset=test_dataset,
        class_name=target_class,
        device=device,
    )

    save_image(
        private_sample.image.detach().cpu(),
        image_dir / f"private_{target_class}.png",
    )

    print("\nPrivate sample:")
    print(f"class={private_sample.class_name}")
    print(f"index={private_sample.dataset_index}")
    print(f"image shape={tuple(private_sample.image.shape)}")
    print(f"label={private_sample.label.item()}")

    # 3. Server creates global model
    global_model = ViTAdapterClassifier(
        backbone_name="vit_tiny_patch16_224",
        num_classes=10,
        bottleneck_dim=32,
        pretrained=True,
    ).to(device)

    print("\nBefore calibration:")
    print(get_trainable_parameter_status(global_model))

    # 4. Calibrate adapter + classifier on public train subset
    print("\nStarting calibration training...")
    global_model = calibrate_adapter_classifier(
        model=global_model,
        train_dataset=train_dataset,
        device=device,
        num_samples=1000,
        batch_size=32,
        epochs=2,
        lr=1e-3,
    )

    print("\nAfter calibration:")
    print(get_trainable_parameter_status(global_model))

    # Optional: save calibrated global model checkpoint
    ckpt_path = checkpoint_dir / "calibrated_vit_adapter.pt"
    torch.save(global_model.state_dict(), ckpt_path)
    print(f"\nSaved calibrated model checkpoint to: {ckpt_path}")

    # 5. Create server with calibrated global model
    server = FLServer(global_model=global_model)

    # 6. Server sends calibrated model copies to two clients
    client_1 = FLClient(
        client_id=1,
        model=server.send_model_to_client(),
        device=device,
    )

    client_2 = FLClient(
        client_id=2,
        model=server.send_model_to_client(),
        device=device,
    )

    print(f"\nCreated Client {client_1.client_id} and Client {client_2.client_id}")

    # 7. Client 1 computes adapter gradient from real private image
    adapter_grads = client_1.compute_adapter_gradient(
        private_x=private_sample.image,
        private_y=private_sample.label,
    )

    print("\nClient 1 adapter gradients after calibration:")
    for i, grad in enumerate(adapter_grads):
        print(f"grad {i}: shape={tuple(grad.shape)}, norm={grad.norm().item():.6f}")

    assert len(adapter_grads) > 0
    assert all(g is not None for g in adapter_grads)

    print("\nStep 3 passed.")
    print("Calibrated global model can produce adapter gradients from held-out private CIFAR-10 image.")


if __name__ == "__main__":
    main()