from .LossFunctionInterface import LossFunctionInterface

class MeanSquaredError(LossFunctionInterface):
    
    @staticmethod
    def loss(a: list, y: list) -> float:
        return 1/(2*len(a))*sum((a[i] - y[i][0])**2 for i in range(len(a)))
    
    @staticmethod
    def derivative(a: list, y: list) -> list:
        return [a[i] - y[i] for i in range(len(a))]