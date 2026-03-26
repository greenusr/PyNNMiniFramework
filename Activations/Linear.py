from .ActivationInterface import ActivationInterface

class Linear(ActivationInterface):
    @staticmethod
    def activation(z: list) -> list:
        return z
    
    @staticmethod
    def activation_derivative(z: list) -> list:
        return [[1 for _ in z]]