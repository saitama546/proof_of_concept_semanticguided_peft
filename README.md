This repository implements a toy semantic-assisted adapter-gradient reconstruction experiment.

The setup contains one server and two simulated clients. The server holds a pretrained ViT backbone with a small trainable adapter. The backbone is frozen and does not contain the clients’ private CIFAR-10 images. One held-out CIFAR-10 image is treated as the private data of Client 1. Client 1 computes an adapter gradient, and this gradient is exposed to the attacker.

The attacker reconstructs the private image under four settings:
1. adapter gradient only,
2. adapter gradient + correct semantic prompt,
3. adapter gradient + incorrect semantic prompt,
4. text prompt only.

CLIP is used only on the attacker side as a semantic prior. The goal is to test whether semantic guidance improves reconstruction from adapter gradients or merely produces prompt-driven hallucinations.

| File                       | What it does                                                                    |
| -------------------------- | ------------------------------------------------------------------------------- |
| `src/model.py`             | Builds the global model: pretrained ViT backbone + adapter + classifier         |
| `src/fl.py`                | Creates `Server` and `Client` classes                                           |
| `src/utils.py`             | Device and seed helper functions                                                |
| `scripts/test_fl_setup.py` | Runs a small test to check that the server/client/model/gradient pipeline works |