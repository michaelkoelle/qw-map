from abc import abstractmethod
from typing import List, Sized, Tuple

import torch
import torch.multiprocessing
from numpy import float32
from torch.utils.data import Dataset


class SizedDataset(Dataset[Tuple[torch.Tensor, float32]], Sized):
    """Base class for Datasets"""

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float32]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def classes(self) -> List[int]:
        """Classes of the dataset"""

    @abstractmethod
    def num_features(self) -> int:
        """Number of features"""

    @abstractmethod
    def class_names(self) -> List[str]:
        """Returns the names of the classes"""

    @abstractmethod
    def feature_names(self) -> List[str]:
        """Returns the names of the features"""

    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes())
