from .ActivationInterface import ActivationInterface
import math

class Sigmoid(ActivationInterface):
    @staticmethod
    def activation(z: list) -> list:
        return [1/(1+math.exp(-x)) for x in z]
    
    @staticmethod
    def activation_derivative(z: list) -> list:
        s = Sigmoid.activation(z)
        return [[s[z_i]*(1-s[z_i]) for z_i in range(len(z))]]
    
    
    