"""Config"""
from dataclasses import dataclass
from enum import Enum
from math import ceil, log2
from typing import Callable, Type

import pennylane as qml
import torch
from pennylane.operation import Operation
from torch import Tensor, nn

from datasets.breast_cancer_dataset import BreastCancerDataset
from datasets.dataset import SizedDataset
from datasets.iris_dataset import IrisDataset, IrisDataset2D
from datasets.wine_dataset import WineDataset
from factories.encoding_factory import EncodingFactory


class Dataset(Enum):
    """Availiable Datasets"""

    IRIS = "iris"
    IRIS_2D = "iris_2d"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"

    @staticmethod
    def get_instance(dataset: str) -> SizedDataset:
        """Get the instance from enum"""
        if dataset == Dataset.IRIS.value:
            return IrisDataset()
        if dataset == Dataset.IRIS_2D.value:
            return IrisDataset2D()
        if dataset == Dataset.WINE.value:
            return WineDataset()
        if dataset == Dataset.BREAST_CANCER.value:
            return BreastCancerDataset()

        raise Exception(f"No such dataset: {dataset}!")


class Encoding(Enum):
    """Availiable Encodings"""

    AMPLITUDE_ENCODING = "amplitude_encoding"
    ANGLE_ENCODING_X = "angle_encoding_x"
    ANGLE_ENCODING_Y = "angle_encoding_y"
    ANGLE_ENCODING_Z = "angle_encoding_z"

    @staticmethod
    def get_instance(encoding: str) -> EncodingFactory:
        """Get the instance from enum"""
        if encoding == Encoding.AMPLITUDE_ENCODING.value:
            return EncodingFactory(qml.AmplitudeEmbedding, pad_with=0.3, normalize=True)
        if encoding == Encoding.ANGLE_ENCODING_X.value:
            return EncodingFactory(qml.AngleEmbedding, rotation="X")
        if encoding == Encoding.ANGLE_ENCODING_Y.value:
            return EncodingFactory(qml.AngleEmbedding, rotation="Y")
        if encoding == Encoding.ANGLE_ENCODING_Z.value:
            return EncodingFactory(qml.AngleEmbedding, rotation="Z")

        raise Exception(f"No such encoding: {encoding}!")

    @staticmethod
    def get_num_qubits(encoding: str, num_classes: int, num_features: int) -> int:
        """Get num qubits"""
        if encoding == Encoding.AMPLITUDE_ENCODING.value:
            return max(num_classes, ceil(log2(num_features)))
        if encoding == Encoding.ANGLE_ENCODING_X.value:
            return max(num_classes, num_features)
        if encoding == Encoding.ANGLE_ENCODING_Y.value:
            return max(num_classes, num_features)
        if encoding == Encoding.ANGLE_ENCODING_Z.value:
            return max(num_classes, num_features)

        raise Exception(f"No such encoding: {encoding}!")


class Layers(Enum):
    """Availiable Layers"""

    STRONGLY_ENTANGLING_LAYERS = "strongly_entangling_layers"
    BASIC_ENTANGLER_LAYERS = "basic_entangler_layers"

    @staticmethod
    def get_instance(layers: str) -> Type[Operation]:
        """Get the instance from enum"""
        if layers == Layers.STRONGLY_ENTANGLING_LAYERS.value:
            return qml.StronglyEntanglingLayers
        if layers == Layers.BASIC_ENTANGLER_LAYERS.value:
            return qml.BasicEntanglerLayers

        raise Exception(f"No such layers: {layers}!")


class ShapingFunction(Enum):
    """Availiable Activations"""

    TANH = "tanh"
    HARD_CLAMP = "hard_clamp"
    SIGMOID = "sigmoid"
    ELU = "elu"
    ARCTAN = "arctan"
    NONE = "none"

    @staticmethod
    def get_instance(activations: str) -> Callable[[Tensor], Tensor]:
        """Get the instance from enum"""
        if activations == ShapingFunction.TANH.value:
            return lambda x: torch.pi * (nn.Tanh())(x)
        if activations == ShapingFunction.HARD_CLAMP.value:
            return lambda x: torch.max(
                torch.min(x, torch.tensor(torch.pi)), torch.tensor(-torch.pi)
            )
        if activations == ShapingFunction.SIGMOID.value:
            return lambda x: 2 * torch.pi * (1 / (1 + torch.pow(torch.e, -x))) - torch.pi
        if activations == ShapingFunction.ELU.value:
            return nn.ELU(torch.pi)
        if activations == ShapingFunction.ARCTAN.value:
            return lambda x: 2.0 * torch.arctan(2 * x)
        if activations == ShapingFunction.NONE.value:
            return lambda x: x

        raise Exception(f"No such activations: {activations}!")


@dataclass
class Config:
    """Config"""

    model: str
    shaping_function: str
    dataset: str
    seed_value: int
    lr: float
    epochs: int
    batch_size: int
    encoding: str
    layers: str
    num_layers: int
    weight_decay: float
    batch_norm: bool
    data_reuploading: bool
