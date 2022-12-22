from pathlib import Path
from typing import List

import numpy as np
from numpy import genfromtxt

from datasets.dataset import SizedDataset


class WineDataset(SizedDataset):
    """Wine dataset"""

    def __init__(self, data: Path = Path("./src/data/wine.txt")) -> None:
        self.data = genfromtxt(data, delimiter=",")
        self.x = self.data[:, 1:]
        self.y = self.data[:, 0]

    def __getitem__(self, index: int):
        return self.x[index].astype(np.float32), np.float32(self.classes().index(self.y[index]))

    def __len__(self):
        return self.x.shape[0]

    def classes(self) -> List[int]:
        return [1, 2, 3]

    def class_names(self) -> List[str]:
        return ["class_0", "class_1", "class_2"]

    def feature_names(self) -> List[str]:
        return [
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]

    def num_features(self) -> int:
        return 13
