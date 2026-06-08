# src/utils.py
#Purpose: Utility functions for the project.
import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """
    Description: Set the random seed for reproducibility.
    INPUTS:
    - seed (int): The random seed to use for reproducibility. Default is 0.
    OUTPUTS:    - None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Description: Get the device to use for PyTorch computations (GPU if available, otherwise CPU).
    INPUTS:    - None
    OUTPUTS:    - device (torch.device): The device to use for PyTorch computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")