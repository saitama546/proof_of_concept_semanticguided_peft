# src/train.py
#purpose : This module contains functions for training the adapter and classifier head of a ViT-based model in a federated learning setup.
# The main function, calibrate_adapter_classifier, takes a model, training dataset, and training parameters to perform
# a brief training loop. The backbone of the model is frozen, while the adapter and classifier are trained on a small public 
# calibration subset of the training data. The function returns the calibrated model
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def get_trainable_parameter_status(model: torch.nn.Module) -> Dict[str, int]:
    """
    Description: Get the number of trainable parameters in each part of the model (backbone, adapter, classifier). This function is useful 
    for verifying that the backbone is frozen and that the adapter and classifier are trainable before starting the training loop.
    INPUTS:
    - model: The PyTorch model for which to check the trainable parameters, expected to be an instance of ViTAdapterClassifier or a 
             similar architecture with backbone, adapter, and classifier components. datatype : torch.nn.Module
    OUTPUT: A dictionary containing the count of trainable parameters for each part of the model, with keys "backbone_trainable_tensors", 
            "adapter_trainable datatype: dict[str, int]_tensors", and "classifier_trainable_tensors". 
    """
    return {
        "backbone_trainable_tensors": sum(
            p.requires_grad for p in model.backbone.parameters()
        ),
        "adapter_trainable_tensors": sum(
            p.requires_grad for p in model.adapter.parameters()
        ),
        "classifier_trainable_tensors": sum(
            p.requires_grad for p in model.classifier.parameters()
        ),
    }


def make_calibration_subset(
    dataset: Dataset,
    num_samples: int = 1000,
) -> Subset:
    """
    Description: Create a subset of the given dataset for calibration training.The function takes the original dataset 
    and the number of samples to include in the subset, and returns a Subset object containing the specified number of 
    samples from the original dataset. This subset will be used for briefly training the adapter and classifier head 
    while keeping the backbone frozen.
    INPUTS:
    - dataset: The original dataset from which to create the calibration subset, expected to be a PyTorch 
        Dataset object. datatype : torch.utils.data.Dataset
    - num_samples: The number of samples to include in the calibration subset (default is 1000). datatype : int
    OUTPUT: A Subset object containing the specified number of samples from the original dataset, 
    which will be used for calibration training. datatype : torch.utils.data.Subset
    """
    indices = list(range(min(num_samples, len(dataset))))
    return Subset(dataset, indices)


def calibrate_adapter_classifier(
    model: torch.nn.Module,
    train_dataset: Dataset,
    device: torch.device,
    num_samples: int = 1000,
    batch_size: int = 32,
    epochs: int = 2,
    lr: float = 1e-3,
) -> torch.nn.Module:
    """
   Description: Calibrate the adapter and classifier head of the model on a small public subset of the training data while keeping the backbone frozen.
   INPUTS:
   - model: The PyTorch model to be calibrated, expected to be an instance of ViTAdapterClassifier or a similar architecture with backbone, adapter, and classifier components. datatype : torch.nn.Module
   - train_dataset: The training dataset to use for calibration, expected to be a PyTorch Dataset object. datatype : torch.utils.data.Dataset
   - device: The torch.device on which to perform the calibration training (e.g., "cuda" or "cpu"). datatype : torch.device
   - num_samples: The number of samples from the training dataset to use for calibration (default is 1000). datatype : int
   - batch_size: The batch size to use for the DataLoader during calibration training (default is 32). datatype : int
   - epochs: The number of epochs to train the adapter and classifier head during calibration (default is 2). datatype : int
   - lr: The learning rate to use for the optimizer during calibration training (default is 1e-3). datatype : float
   OUTPUT: The calibrated model with the adapter and classifier head trained on the specified subset of the training data, while the backbone remains frozen. datatype : torch.nn.Module
    """
    model.to(device)
    model.train()

    # Ensure backbone is frozen.
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Ensure adapter and classifier are trainable.
    for param in model.adapter.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    calibration_subset = make_calibration_subset(
        dataset=train_dataset,
        num_samples=num_samples,
    )

    loader = DataLoader(
        calibration_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        list(model.adapter.parameters()) + list(model.classifier.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        print(
            f"Calibration epoch {epoch + 1}/{epochs} "
            f"| loss={avg_loss:.4f} | acc={acc:.4f}"
        )

    return model