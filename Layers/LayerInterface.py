from abc import ABC, abstractmethod
from Activations.ActivationFactory import ActivationFatory

class LayerInterface(ABC):
    
    def __init__(self, input_size: int, output_size: int, activation_name: str, lambda_: float):
        self.input_size = input_size
        self.output_size = output_size
        
        self.activation_name = activation_name
        self.activation, self.activation_derivative = ActivationFatory.get_activation(activation_name)
        
        self.lambda_ = lambda_
        
    
    @abstractmethod
    def forward(self, a: list) -> list:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, da: list, learning_rate: float) -> list:
        raise NotImplementedError
    