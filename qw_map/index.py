"""
Quantum Weight Re-Mapping Functions
"""
import torch


def none(x: torch.Tensor) -> torch.Tensor:
    """
    Identity function
    """
    return x


def clamp(x: torch.Tensor) -> torch.Tensor:
    """
    Clamp function. Maps values above \\pi to \\pi and below -\\pi to -\\pi.
    """
    return torch.max(torch.min(x, torch.tensor(torch.pi)), torch.tensor(-torch.pi))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid function. Scaled appropriatly to interval [-\\pi;\\pi].
    """
    return 2 * torch.pi * (1 / (1 + torch.pow(torch.e, -x))) - torch.pi


def arctan(x: torch.Tensor) -> torch.Tensor:
    """
    Arcus Tangent function. Scaled appropriatly to interval [-\\pi;\\pi].
    """
    return 2.0 * torch.arctan(2 * x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Tangent Hyperbolicus. Scaled appropriatly to interval [-\\pi;\\pi]
    """
    return torch.pi * torch.tanh(x)
