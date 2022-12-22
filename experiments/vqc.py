# Workaround because Any != Any
# pyright: reportIncompatibleMethodOverride=false

from typing import Callable, List, Type

import pennylane as qml
import torch
from pennylane.operation import Operation
from torch import Tensor, nn

from factories.encoding_factory import EncodingFactory


class VQC(nn.Module):
    """Variational circuit"""

    def __init__(
        self,
        classes: List[int],
        num_qubits: int,
        encoding: EncodingFactory,
        layers: Type[Operation],
        shaping_function: Callable[[Tensor], Tensor],
        num_layers: int,
        batch_norm: bool,
        data_reuploading: bool,
    ) -> None:
        super().__init__()
        self.device = qml.device("default.qubit", wires=num_qubits)
        self.classes = classes
        self.batch_norm = batch_norm

        def circuit(ws: Tensor, x: Tensor):
            if data_reuploading:
                for w in ws:
                    encoding.create(x, wires=range(num_qubits))
                    layers(shaping_function(w.unsqueeze(0)), wires=range(num_qubits))
            else:
                encoding.create(x, wires=range(num_qubits))
                layers(shaping_function(ws), wires=range(num_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(len(self.classes))]

        self.qnode = qml.QNode(circuit, self.device, interface="torch")
        self.weights = nn.Parameter(
            0.01 * torch.rand(num_layers, num_qubits, 3), requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(len(self.classes)), requires_grad=True)
        self.bn = nn.BatchNorm1d(len(self.classes), affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """forward pass"""
        res = torch.stack(
            [nn.Softmax()(self.qnode(self.weights, state) + 1 * self.bias) for state in x]
        )

        if self.batch_norm:
            res = self.bn(res.float()).float()
        return res

    def draw(self, x: Tensor):
        """Draws the circuit to console"""
        print(qml.draw(self.qnode)(self.weights, x))
