import os
import sys
import math
import torch
import random
import sklearn
import requests
import datasets
import warnings
import torchvision
import transformers
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torch.nn as nn
from typing import Optional
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_metric
import torchvision.transforms as transforms
from IPython.display import display, Markdown
from torch.utils.data import Dataset, DataLoader


class Constants:
    seed = 42
    batch_size = 32
    lr = 1e-3
    # CLIP models
    clip_base_link = "openai/clip-vit-base-patch32"
    clip_large_link = "openai/clip-vit-large-patch14"
    clip_laion_link = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    # ALIGN model
    align_link = "kakaobrain/align-base"
    # SigLIP models
    siglip_so_link = "google/siglip-so400m-patch14-384"
    siglip_base_link = "google/siglip-base-patch16-224"
    siglip_large_link = "google/siglip-large-patch16-384"
    siglip_large_256_link = "google/siglip-large-patch16-256"
    # AltCLIP model
    altclip_link = "BAAI/AltCLIP"
    # Datasets
    cifar10_link = "Andron00e/CIFAR10-Custom"
    cifar100_link = "Andron00e/CIFAR100-Custom"
    cub200_link = "Andron00e/CUB200-Custom"
    places365_link = "Andron00e/Places365-custom"
    imagenet_link = "imagenet-1k"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(device_no: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def print_centered_text(text):
    display(Markdown(f"<center>{text}</center>"))
