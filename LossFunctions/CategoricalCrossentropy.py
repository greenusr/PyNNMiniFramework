from .LossFunctionInterface import LossFunctionInterface
import math
class CategoricalCrossentropy(LossFunctionInterface):
    
    @staticmethod
    def loss(a: list, y: list) -> float:
        return 1/len(a) * sum(sum(-y[o][i]*math.log(a[o][i]) for i in range(len(a[0]))) for o in range(len(a)))
    
    @staticmethod
    def derivative(a: list, y: list) -> list:
        return [-y[i]/a[i] for i in range(len(a))]