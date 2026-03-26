from .ActivationInterface import ActivationInterface
from typing import Callable

from .Sigmoid import Sigmoid
from .Linear import Linear
from .Softmax import Softmax
from .Relu import Relu

class ActivationFatory:
    @staticmethod
    def get_activation(activation_name: str) -> tuple[ Callable[[list], list], Callable[[list],list]]:
        
        activation_dict: dict[
            str,
            tuple[ Callable[[list], list], Callable[[list],list]]
        ] = {
            "sigmoid": (Sigmoid.activation, Sigmoid.activation_derivative),
            "linear": (Linear.activation, Linear.activation_derivative),
            "softmax": (Softmax.activation, Softmax.activation_derivative),
            "relu": (Relu.activation, Relu.activation_derivative)
        }
        
        return activation_dict.get(activation_name)