"""Classical neural networks"""
# Workaround because Any != Any
# pyright: reportIncompatibleMethodOverride=false
import math
from typing import List

from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """Neural Network for VQC comparison"""

    def __init__(
        self,
        classes: List[int],
        num_input: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_output = len(classes)

        # num_QVC_params = layers * inputs * 3 + output
        self.num_params_of_QVC = num_layers * self.num_input * 3 + self.num_output

        # num_QVC_params = input*hidden1+hidden1 + hidden1*hidden2+hidden2  + hidden2*output + output
        # we define and set hidden := hidden1 = hidden2
        # Q = i*h+h + h*h+h + h*o+o --> h^2 + h*(i+o+2) + o - Q = 0
        # h = {-(i+o+2) + sqrt[(i+o+2)^2 - 4(o-Q)]}/2
        # I approximate it with math.ceil to give the max number of neurons closer to the QVC one
        self.num_hidden = math.ceil(
            (
                -(self.num_input + self.num_output + 2)
                + math.sqrt(
                    (self.num_input + self.num_output + 2) ** 2
                    - 4 * (self.num_output - self.num_params_of_QVC)
                )
            )
            / 2
        )
        print(
            "Num layers: {}, num inputs: {}, num outputs: {}".format(
                num_layers, self.num_input, self.num_output
            )
        )
        print("Number of parameters of QVC: ", self.num_params_of_QVC)
        print("Dimensions: ", self.num_input, self.num_hidden, self.num_hidden, self.num_output)
        print(
            "Total params: ",
            self.num_input * self.num_hidden
            + self.num_hidden
            + self.num_hidden * self.num_hidden
            + self.num_hidden
            + self.num_hidden * self.num_output
            + self.num_output,
        )

        self.model = nn.Sequential(
            # nn.Linear(self.num_input, self.num_hidden),
            # nn.ELU(),
            # nn.Linear(self.num_hidden, self.num_hidden),
            # nn.ELU(),
            # nn.Linear(self.num_hidden, self.num_output),
            # nn.Sigmoid(),
            nn.Linear(self.num_input, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, self.num_output),
            nn.Sigmoid(),
        )
        # self.double()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        return self.model(x)
