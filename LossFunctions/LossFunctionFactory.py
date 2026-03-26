from .LossFunctionInterface import LossFunctionInterface
from typing import Callable

from .MeanSquaredError import MeanSquaredError
from .BinaryCrossentropy import BinaryCrossentropy
from .CategoricalCrossentropy import CategoricalCrossentropy

class LossFunctionFactory:
    @staticmethod
    def get_loss_function(loss_function_name: str) ->tuple[Callable[[list,list], float],Callable[[list,list],list]]:
        loss_function_dict: dict[
            str,
            tuple[Callable[[list,list],float], Callable[[list,list],list]]
        ] = {
            "mean_squared_error": (MeanSquaredError.loss, MeanSquaredError.derivative),
            "binary_crossentropy": (BinaryCrossentropy.loss, BinaryCrossentropy.derivative),
            "categorical_crossentropy": (CategoricalCrossentropy.loss, CategoricalCrossentropy.derivative),
        }
        return loss_function_dict.get(loss_function_name)