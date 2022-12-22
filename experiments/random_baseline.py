"""Random baseline"""
# Workaround because Any != Any
# pyright: reportIncompatibleMethodOverride=false
from typing import List

import torch
from torch import Tensor, nn


class RandomBaseline(nn.Module):
    """Random Baseline for VQC comparison"""

    def __init__(
        self,
        classes: List[int],
    ) -> None:
        super().__init__()
        self.num_output = len(classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        return torch.rand(x.shape[0], self.num_output)
