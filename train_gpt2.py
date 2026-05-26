from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """
    Configuration class for the GPT-2 model.
    Defines the hyperparameters used to initialize the model architecture.
    """
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):
    """
    The GPT-2 language model architecture.
    Subclasses torch.nn.Module to leverage PyTorch's neural network features.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config