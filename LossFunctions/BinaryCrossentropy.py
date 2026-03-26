from .LossFunctionInterface import LossFunctionInterface
import math
class BinaryCrossentropy(LossFunctionInterface):
    
    @staticmethod
    def loss(a: list, y: list) -> float:
        total_loss = 0
        for s_i in range(len(a)):
            total_loss+=-y[s_i][0]*math.log(a[s_i][0]) - (1-y[s_i][0])*math.log(1-a[s_i][0])
        return 1/len(a) * total_loss
    
    @staticmethod
    def derivative(a: list, y: list) -> list:
        return [-(y[i]/a[i] - (1-y[i])/(1-a[i])) for i in range(len(a))]