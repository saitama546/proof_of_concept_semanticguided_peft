# scripts/test_real_cifar_gradient.py

import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import load_cifar10, get_private_sample, split_dataset_two_clients
from src.model import ViTAdapterClassifier
from src.fl import FLServer, FLClient
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

    # Output folder
    image_dir = ROOT / "results" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 2. Create two simulated clients from train split
    client_datasets = split_dataset_two_clients(
        dataset=train_dataset,
        num_samples_per_client=1000,
    )

    print(f"Client 1 dataset size: {len(client_datasets[1])}")
    print(f"Client 2 dataset size: {len(client_datasets[2])}")

    # 3. Select one held-out private image from test split
    target_class = "frog"

    private_sample = get_private_sample(
        dataset=test_dataset,
        class_name=target_class,
        device=device,
    )

    print("\nPrivate sample selected:")
    print(f"Class name: {private_sample.class_name}")
    print(f"Dataset index: {private_sample.dataset_index}")
    print(f"Image shape: {tuple(private_sample.image.shape)}")
    print(f"Label tensor: {private_sample.label}")

    save_path = image_dir / f"private_{target_class}.png"
    save_image(private_sample.image.detach().cpu(), save_path)
    print(f"Saved private image to: {save_path}")

    # 4. Server creates global model
    global_model = ViTAdapterClassifier(
        backbone_name="vit_tiny_patch16_224",
        num_classes=10,
        bottleneck_dim=32,
        pretrained=True,
    ).to(device)

    server = FLServer(global_model=global_model)

    # 5. Server sends model copies to clients
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

    print(f"\nClient {client_1.client_id} and Client {client_2.client_id} created.")

    # 6. Client 1 computes adapter gradient on real private image
    adapter_grads = client_1.compute_adapter_gradient(
        private_x=private_sample.image,
        private_y=private_sample.label,
    )

    print("\nClient 1 adapter gradients from real CIFAR-10 private image:")
    for i, grad in enumerate(adapter_grads):
        print(f"grad {i}: shape={tuple(grad.shape)}, norm={grad.norm().item():.6f}")

    assert len(adapter_grads) > 0
    assert all(g is not None for g in adapter_grads)

    print("\nStep 2 passed.")
    print("Real CIFAR-10 private image can produce adapter gradients.")


if __name__ == "__main__":
    main()