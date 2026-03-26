from .ActivationInterface import ActivationInterface

class Relu(ActivationInterface):
    @staticmethod
    def activation(z: list) -> list:
        return [max(0,x) for x in z]
    
    @staticmethod
    def activation_derivative(z: list) -> list:
        return [[1 if x >0 else 0 for x in z]]