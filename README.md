This repository implements a  semantic-guided adapter-gradient attack mechanism to reconstruct the images  .

The setup contains one server and two simulated clients. The server holds a pretrained ViT backbone with a small trainable adapter. The backbone is frozen and does not contain the clients’ private CIFAR-10 images. One held-out CIFAR-10 image is treated as the private data of Client 1. Client 1 computes an adapter gradient, and this gradient is exposed to the attacker.

The attacker reconstructs the private image under four settings:
1. adapter gradient only,
2. adapter gradient + correct semantic prompt,
3. adapter gradient + incorrect semantic prompt,
4. text prompt only.

