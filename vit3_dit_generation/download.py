# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for loading DiT checkpoints."""

import os

import torch
from torchvision.datasets.utils import download_url


pretrained_models = {"DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"}


def find_model(model_name):
    """Find a pre-trained DiT model or load a local checkpoint.

    If the checkpoint is produced by train.py and contains an EMA state dict,
    this function returns the EMA weights by default.
    """
    if model_name in pretrained_models:
        return download_model(model_name)

    assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def download_model(model_name):
    """Download an official pre-trained DiT checkpoint if needed."""
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = f"https://dl.fbaipublicfiles.com/DiT/models/{model_name}"
        download_url(web_path, "pretrained_models")
    return torch.load(local_path, map_location=lambda storage, loc: storage)


if __name__ == "__main__":
    for model in pretrained_models:
        download_model(model)
    print("Done.")
