import torch
import torch.nn as nn

import torch

def normalize_to_length1(x: torch.Tensor, eps: float = 1e-12):
    """
    x: (batch, channels, length)
    벡터의 방향만 남김
    """
    norm = (x.pow(2).sum(dim=1, keepdim=True).sqrt()).clamp_min(eps)
    return x / norm

#긴거 자르는 함수

#