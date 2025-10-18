from __future__ import annotations
import torch

def compute_tensor_summary(input_tensor: torch.Tensor) -> dict[str, float]:
    """
    Return simple summary statistics of a tensor as Python floats.
    Useful for quick checks in notebooks.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Any numeric tensor.

    Returns
    -------
    dict[str, float]
        Dictionary with mean, standard_deviation, minimum, maximum.
    """
    return {
        "mean": float(input_tensor.mean().item()),
        "standard_deviation": float(input_tensor.std(unbiased=False).item()),
        "minimum": float(input_tensor.min().item()),
        "maximum": float(input_tensor.max().item()),
    }
