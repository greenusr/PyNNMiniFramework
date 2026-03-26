from abc import ABC, abstractmethod

class ActivationInterface(ABC):
    
    @staticmethod
    @abstractmethod
    def activation(z: list) -> list:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def activation_derivative(z: list) -> list:
        raise NotImplementedError