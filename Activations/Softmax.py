from .ActivationInterface import ActivationInterface
import math

class Softmax(ActivationInterface):
    @staticmethod
    def activation(z: list) -> list:
        return [math.exp(x_o)/sum(math.exp(x_i) for x_i in z) for x_o in z]
    
    @staticmethod
    def activation_derivative(z: list) -> list:
        jacobian = []
        s = Softmax.activation(z)
        for r_i in range(len(z)):
            row = []
            for c_i in range(len(z)):
                row.append(s[r_i]*(1 -s[r_i]) if r_i == c_i else -s[r_i]*s[c_i])
            jacobian.append(row)
        return jacobian