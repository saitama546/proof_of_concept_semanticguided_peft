# Purpose: To build the global model used by the server, which consists of a pretrained ViT backbone (frozen),
# a small trainable adapter, and a classifier head. The adapter allows for fine-tuning on 
# new tasks while keeping the backbone fixed, and the classifier head maps the adapted features to class  logits.


from __future__ import annotations
from ast import Global
from typing import List
import torch
import torch.nn as nn
import timm


class BottleneckAdapter(nn.Module):
    """
    Description: A small adapter module that can be inserted into the global Server ViT backbone.
    INPUT: feature_dim (dimension of features from the ViT backbone) datatype: int
    OUTPUT: adapted features of the same dimension  datatype: torch.Tensor
    ARGS - bottleneck_dim: dimension of the bottleneck layer datatype: int, default=32
    """

    def __init__(self, feature_dim: int, bottleneck_dim: int = 32):
        """
        Description: Initialize the adapter with a down-projection, activation, and up-projection
        INPUT: feature_dim (dimension of features from the ViT backbone) datatype: int
               bottleneck_dim (dimension of the bottleneck layer) datatype: int, default=32
        OUTPUT: None
        """
        super().__init__()
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description: Forward pass through the adapter    
        INPUT: x (input features from the ViT backbone) datatype: torch.Tensor
        OUTPUT: adapted features of the same dimension as input datatype: torch.Tensor
        """
        return self.up(self.activation(self.down(x)))


class ViTAdapterClassifier(nn.Module):
    """
    Description: A global model that combines a frozen pretrained ViT backbone, a trainable adapter, and a classifier head.
    INPUTS:
    - backbone_name: name of the ViT model to use from timm (e.g., "vit_tiny_patch16_224") datatype: str
    - num_classes: number of output classes for the classifier head datatype: int, default=10
    - bottleneck_dim: dimension of the adapter bottleneck layer datatype: int, default=32
    - pretrained: whether to load pretrained weights for the ViT backbone datatype: bool, default=True
    OUTPUT: logits for each class datatype: torch.Tensor
    """
    

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        num_classes: int = 10,
        bottleneck_dim: int = 32,
        pretrained: bool = True,
    ):
        """
        Description: Initialize the global model with a pretrained ViT backbone, a bottleneck adapter, and a classifier head.
        INPUTS:
        - backbone_name: name of the ViT model to use from timm (e.g., "vit_tiny_patch16_224") datatype: str
        - num_classes: number of output classes for the classifier head datatype: int, default=10
        - bottleneck_dim: dimension of the adapter bottleneck layer datatype: int, default=32
        - pretrained: whether to load pretrained weights for the ViT backbone datatype: bool, default=True
        OUTPUT: None    
        """
        super().__init__()

        # Load pretrained ViT and remove its original classifier.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        self.adapter = BottleneckAdapter(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """
        Description: Freeze the backbone parameters so they are not updated during training. This ensures that only the adapter and classifier head are trained, while the ViT backbone remains fixed. This is important for our attack scenario, as we want to observe gradients only from the adapter and classifier head.
        INPUT: None
        OUTPUT: None
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_adapter_parameters(self) -> List[nn.Parameter]:
        """
        Description: Get the parameters of the adapter and classifier head that will be updated during training. This is important for our attack scenario, as
        INPUT: None
        OUTPUT: List of parameters from the adapter and classifier head that will be updated during training.   
        """
        return list(self.adapter.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description: Forward pass through the global model. The input is passed through the frozen ViT backbone to extract features, then through the adapter to get adapted features, and finally through the classifier head to get logits for each class.
        INPUT: x (input images) datatype: torch.Tensor
        OUTPUT: logits for each class datatype: torch.Tensor
        """
        features = self.backbone(x)
        adapted_features = features + self.adapter(features)
        logits = self.classifier(adapted_features)
        return logits
