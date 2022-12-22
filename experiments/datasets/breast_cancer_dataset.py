from pathlib import Path
from typing import List

import numpy as np

from datasets.dataset import SizedDataset


class BreastCancerDataset(SizedDataset):
    """Cancer dataset"""

    def __init__(self, data: Path = Path("./src/data/cancer.txt")) -> None:
        self.data = np.loadtxt(data)

        self.x = self.data[:, :30]
        self.y = self.data[:, -1]

    def __getitem__(self, index: int):
        y = self.y[index].astype(np.float32)
        return self.x[index].astype(np.float32), y if y == 1.0 else np.float32(0.0)

    def __len__(self):
        return self.x.shape[0]

    def num_features(self) -> int:
        return 30

    def classes(self) -> List[int]:
        return [0, 1]

    def class_names(self) -> List[str]:
        return ["Malignant", "Benign"]

    def feature_names(self) -> List[str]:
        return [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
            "mean symmetry",
            "mean fractal dimension",
            "radius error",
            "texture error",
            "perimeter error",
            "area error",
            "smoothness error",
            "compactness error",
            "concavity error",
            "concave points error",
            "symmetry error",
            "fractal dimension error",
            "worst radius",
            "worst texture",
            "worst perimeter",
            "worst area",
            "worst smoothness",
            "worst compactness",
            "worst concavity",
            "worst concave points",
            "worst symmetry",
            "worst fractal dimension",
        ]
