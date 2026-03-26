from abc import ABC, abstractmethod

class LossFunctionInterface(ABC):
    
    @staticmethod
    @abstractmethod
    def loss(a: list, y: list) -> float:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def derivative(a: list, y: list) -> list:
        raise NotImplementedError