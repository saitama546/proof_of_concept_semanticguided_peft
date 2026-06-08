# src/clip_prior.py

from __future__ import annotations

import torch
import torch.nn.functional as F
import open_clip


class CLIPTextPrior:
    """
    Description:
        CLIP-based text-image similarity loss for image generation. 
    INPUTS: 
        device: The device to perform computations on (e.g., torch.device("cuda")).
        model_name: The name of the CLIP model to use.
        pretrained: The pretrained weights to use.  
    OUTPUTS:
        A CLIPTextPrior object that can compute the text-image similarity loss.
    METHODS:
        encode_text(prompt: str) -> torch.Tensor:
            Encode a text prompt into a feature vector.
        text_image_loss(image: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
            Compute the CLIP semantic loss between an image and text features.  
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        """
        Description:
            Initialize the CLIPTextPrior object by loading the specified CLIP model and tokenizer.
        Inputs:
            device: The device to perform computations on (e.g., torch.device("cuda")). datatype: torch.device
            model_name: The name of the CLIP model to use (default: "ViT-B-32"). datatype: str
            pretrained: The pretrained weights to use (default: "openai").      datatype: str
        Outputs:
            A CLIPTextPrior object that can compute the text-image similarity loss. datatype: None
        """
        self.device = device

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.model = self.model.to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Description:
            Encode a text prompt into a feature vector using the CLIP model's text encoder.
        Inputs:
            prompt: The text prompt to encode. datatype: str
        Outputs: A normalized feature vector representing the text prompt. datatype: torch.Tensor
        """
        tokens = self.tokenizer([prompt]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def text_image_loss(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Description:
            Compute the CLIP semantic loss between an image and text features.
        Inputs:
            image: The input image tensor. datatype: torch.Tensor
            text_features: The feature vector representing the text prompt. datatype: torch.Tensor
        Outputs:
            The computed CLIP semantic loss. datatype: torch.Tensor
        """
        clip_image = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        image_features = self.model.encode_image(clip_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T

        return 1.0 - similarity.mean()