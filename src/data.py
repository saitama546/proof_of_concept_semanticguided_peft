# src/data.py
#purpose This module provides functions for loading and processing the CIFAR-10 dataset, which is used in the federated learning experiments.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class PrivateSample:
    """
    Description: A simple data structure to hold a single private sample from the CIFAR-10 dataset, including the image, label, class name, and dataset index.
    INPUTS:
    - image: A tensor representing the image data, expected to be of shape [1, 3, H, W] after processing. datatype : torch.Tensor
    - label: A tensor containing the integer label for the image, expected to be of shape [1]. datatype : torch.Tensor
    - class_name: The string name of the class corresponding to the label (e.g., "frog"). datatype : str
    - dataset_index: The index of the sample in the original CIFAR-10 dataset, useful for reference and debugging. datatype : int
    OUTPUT: An instance of PrivateSample containing the provided information. datatype : PrivateSample
    """
    image: torch.Tensor      # [1, 3, H, W]
    label: torch.Tensor      # [1]
    class_name: str
    dataset_index: int


def get_cifar10_transform(image_size: int = 224):
    """
    Description: Get the transformation pipeline for CIFAR-10 images, including resizing and conversion to tensor.
    INPUTS:
    image_size: The desired size to which CIFAR-10 images will be resized (default is 224). datatype : int
    OUTPUT: A torchvision.transforms.Compose object that defines the transformation pipeline for CIFAR-10 images, including resizing to the specified 
            image size and conversion to tensor. 
            datatype : torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_cifar10(
    root: str = "data",
    train: bool = True,
    image_size: int = 224,
    download: bool = True,
) -> Dataset:
    """
    Description: Load the CIFAR-10 dataset with the specified transformations. The function allows for loading either the training 
    or test split of the dataset, applies resizing and tensor conversion to the images, and optionally downloads the dataset if it 
    is not already present in the specified root directory.
    INPUTS:
    - root: The directory where the CIFAR-10 dataset will be stored or is already located (default is "data"). datatype : str
    - train: A boolean indicating whether to load the training split (True) or the test split (False) of the CIFAR-10 dataset (default is True). datatype : bool
    - image_size: The desired size to which CIFAR-10 images will be resized (default is 224). datatype : int
    - download: A boolean indicating whether to download the CIFAR-10 dataset if it is not already present in the specified root directory (default is True). datatype : bool
    OUTPUT: A PyTorch Dataset object containing the CIFAR-10 data with the specified transformations applied.
    The dataset will consist of either the training or test split based on the input parameters. datatype : torch.utils.data.Dataset
    """
    transform = get_cifar10_transform(image_size=image_size)

    return datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def class_name_to_label(class_name: str) -> int:
    """
    Description: Convert a CIFAR-10 class name to its corresponding integer label. The function checks
    if the provided class name is valid and returns the index of the class in the CIFAR10_CLASSES list, 
    which corresponds to the integer label used in the dataset.
    INPUTS:
    - class_name: The name of the CIFAR-10 class to convert (e.g., "frog"). datatype : str
    OUTPUT: The integer label corresponding to the provided class name. If the class name is not valid, the 
    function raises a ValueError. datatype : int
    """
    if class_name not in CIFAR10_CLASSES:
        raise ValueError(
            f"Unknown CIFAR-10 class '{class_name}'. "
            f"Valid classes: {CIFAR10_CLASSES}"
        )

    return CIFAR10_CLASSES.index(class_name)


def label_to_class_name(label: int) -> str:
    """
    Description: Convert a CIFAR-10 integer label to its corresponding class name. 
    The function checks if the provided label is valid and returns the class name from the CIFAR10_CLASSES 
    list based on the label index.
    INPUTS:
    - label: The integer label to convert, expected to be in the range [0, 9] for CIFAR-10 (e.g., 6 for "frog"). datatype : int
    OUTPUT: The string class name corresponding to the provided integer label. If the label is not valid, the function raises a ValueError. datatype : str
    """
    return CIFAR10_CLASSES[label]


def find_first_sample_by_class(
    dataset: Dataset,
    class_name: str,
) -> Tuple[int, torch.Tensor, int]:
    """
    Description: Search through the given dataset to find the first sample that belongs to the specified class name. The function converts the class name to its corresponding integer label.
    INPUTS:
    - dataset: The dataset through which to search, expected to be a PyTorch Dataset object. datatype : torch.utils.data.Dataset
    - class_name: The name of the class for which to find a sample (e.g., "frog"). datatype : str
    OUTPUT: A tuple containing the index of the found sample in the dataset, the image tensor, and the integer label. The image tensor is expected to be of shape [1, 3, H, W] after processing.
    datatype : Tuple[int, torch.Tensor, int]
    """
    target_label = class_name_to_label(class_name)

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if label == target_label:
            return idx, image, label

    raise RuntimeError(f"No sample found for class '{class_name}'.")


def get_private_sample(
    dataset: Dataset,
    class_name: str,
    device: torch.device,
) -> PrivateSample:
    """
    Description: Retrieve a single private sample from the dataset that belongs to the specified class. The function searches through the dataset to find the first sample that matches the given class name, converts the image and label to tensors, and returns them in a structured format as a PrivateSample instance.
    INPUTS:
    - dataset: The dataset from which to retrieve the private sample, expected to be a PyTorch Dataset object. datatype : torch.utils.data.Dataset
    - class_name: The name of the class for which to find a sample (e.g., "frog"). datatype : str
    - device: The torch.device on which to place the returned tensors (e.g., "cuda" or "cpu"). datatype : torch.device
    OUTPUT: An instance of PrivateSample containing the image tensor, label tensor, class name, and dataset index of the found sample. The image tensor is expected to be of
    """
    idx, image, label = find_first_sample_by_class(
        dataset=dataset,
        class_name=class_name,
    )

    image = image.unsqueeze(0).to(device)
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)

    return PrivateSample(
        image=image,
        label=label_tensor,
        class_name=class_name,
        dataset_index=idx,
    )


def split_dataset_two_clients(
    dataset: Dataset,
    num_samples_per_client: int = 1000,
):
    """
    Description: Split a given dataset into two subsets for two clients in a federated learning setup. The function takes a dataset 
    and the number of samples to assign to each client, and returns a dictionary containing the subsets for each client. 
    The splitting is done by assigning even-indexed samples to Client 1 and odd-indexed samples to Client 2, ensuring that each client receives the specified number of samples.
    INPUTS:
    - dataset: The original dataset to be split, expected to be a PyTorch Dataset object. datatype : torch.utils.data.Dataset
    - num_samples_per_client: The number of samples to assign to each client (default is 1000). datatype : int
    OUTPUT: A dictionary containing the subsets for each client, where the keys are client IDs (1 and 2) and the values are Subset objects 
    representing the respective datasets for each client. 
    datatype : Dict[int, torch.utils.data.Subset]
    """
    client_1_indices = list(range(0, len(dataset), 2))[:num_samples_per_client]
    client_2_indices = list(range(1, len(dataset), 2))[:num_samples_per_client]

    return {
        1: Subset(dataset, client_1_indices),
        2: Subset(dataset, client_2_indices),
    }