from pathlib import Path
from typing import List

import numpy as np
from numpy import genfromtxt

from datasets.dataset import SizedDataset


# TODO search for original dataset
class IrisDataset(SizedDataset):
    """Iris dataset"""

    def __init__(self, data: Path = Path("./src/data/iris.txt")) -> None:
        self.data = genfromtxt(data, delimiter=",")
        self.x = self.data[:, 0:4]
        self.y = self.data[:, -1]

    def __getitem__(self, index: int):
        y = self.y[index].astype(np.float32)
        return self.x[index].astype(np.float32), np.float32(self.classes().index(y))

    def __len__(self):
        return self.x.shape[0]

    def num_features(self) -> int:
        return 4

    def classes(self) -> List[int]:
        return [0, 1, 2]

    def class_names(self) -> List[str]:
        return ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]

    def feature_names(self) -> List[str]:
        return [
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ]


# TODO search for original dataset
class IrisDataset2D(SizedDataset):
    """Iris dataset reduced dim=2"""

    def __init__(self, data: Path = Path("./src/data/iris.txt")) -> None:
        self.data = genfromtxt(data, delimiter=",")
        self.x = self.data[:, 0:2]
        self.y = self.data[:, -1]

    def __getitem__(self, index: int):
        y = self.y[index].astype(np.float32)
        return self.x[index].astype(np.float32), np.float32(self.classes().index(y))

    def __len__(self):
        return self.x.shape[0]

    def num_features(self) -> int:
        return 2

    def classes(self) -> List[int]:
        return [-1, 1]

    def class_names(self) -> List[str]:
        return ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]  # TODO

    def feature_names(self) -> List[str]:
        return [
            "petal length",
            "petal width",
        ]
