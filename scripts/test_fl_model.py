# scripts/test_fl_setup.py
#purpose This script tests the basic setup of a federated learning (FL) system with a server and two clients. 
# It verifies that the server can create a global model, send it to clients, and that clients can compute 
# gradients for their adapters while keeping the backbone frozen. The script uses dummy data for testing and 
# includes assertions to ensure the expected behavior.

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model import ViTAdapterClassifier
from src.fl import FLServer, FLClient
from src.utils import set_seed, get_device


def main():
    """
    Description: Test the basic setup of a federated learning system with a server and two clients.
    The server creates a global model and sends it to two clients. Each client computes adapter gradients
    using dummy private data. The script checks that the backbone is frozen and that adapter gradients are computed correctly.  
    INPUTS: None (uses dummy data for testing)
    OUTPUT:     - Prints the status of the server and clients, including model parameter statuses and adapter gradients.
                - Assertions to verify that the backbone is frozen and adapter gradients are computed.  
    """
    set_seed(0)
    device = get_device()

    print(f"Using device: {device}")

    # 1. Server creates global model.
    global_model = ViTAdapterClassifier(
        backbone_name="vit_tiny_patch16_224",
        num_classes=10,
        bottleneck_dim=32,
        pretrained=True,
    ).to(device)

    server = FLServer(global_model=global_model)

    # 2. Server sends model copies to two clients.
    client_1_model = server.send_model_to_client()
    client_2_model = server.send_model_to_client()

    client_1 = FLClient(
        client_id=1,
        model=client_1_model,
        device=device,
    )

    client_2 = FLClient(
        client_id=2,
        model=client_2_model,
        device=device,
    )

    # 3. For Step 1, use dummy private data.
    # Later we will replace this with one held-out CIFAR-10 image.
    private_x = torch.randn(1, 3, 224, 224, device=device)
    private_y = torch.tensor([6], dtype=torch.long, device=device)  # pretend frog label

    # 4. Client 1 computes adapter gradient.
    adapter_grads = client_1.compute_adapter_gradient(
        private_x=private_x,
        private_y=private_y,
    )

    # 5. Print checks.
    print("\nServer created global pretrained ViT + adapter model.")
    print("Server sent model copies to Client 1 and Client 2.")

    backbone_trainable = sum(p.requires_grad for p in client_1.model.backbone.parameters())
    adapter_trainable = sum(p.requires_grad for p in client_1.model.adapter.parameters())
    classifier_trainable = sum(p.requires_grad for p in client_1.model.classifier.parameters())

    print("\nClient 1 parameter status:")
    print("Backbone trainable parameter tensors:", backbone_trainable)
    print("Adapter trainable parameter tensors:", adapter_trainable)
    print("Classifier trainable parameter tensors:", classifier_trainable)

    print("\nClient 1 adapter gradients:")
    for i, grad in enumerate(adapter_grads):
        print(f"grad {i}: shape={tuple(grad.shape)}, norm={grad.norm().item():.6f}")

    print(f"\nClient 2 exists with id={client_2.client_id}, but is not attacked in Step 1.")

    # 6. Assertions.
    assert backbone_trainable == 0, "Backbone should be frozen."
    assert adapter_trainable > 0, "Adapter should be trainable."
    assert classifier_trainable > 0, "Classifier should be trainable."
    assert len(adapter_grads) > 0, "Adapter gradients were not computed."
    assert all(g is not None for g in adapter_grads), "Some adapter gradients are None."

    print("\nStep 1 passed.")
    print("Server + two clients + pretrained ViT adapter model + adapter gradient computation works.")


if __name__ == "__main__":
    main()