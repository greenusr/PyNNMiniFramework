from Layers.LayerInterface import LayerInterface
from LossFunctions.LossFunctionFactory import LossFunctionFactory
class Model:
    def __init__(self, layers: list[LayerInterface], loss_function_name: str):
        self.layers = layers
        self.loss_function_name = loss_function_name
        self.loss_function, self.loss_function_derivative = LossFunctionFactory.get_loss_function(loss_function_name)
        
    def forward(self, x: list) -> list:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, da: list, learning_rate: float):
        for layer in reversed(self.layers):
            da = layer.backward(da,learning_rate)
    
    def predict(self, x: list) -> list:
        return self.forward(x)
    
    def fit(self, dataset: list, learning_rate: float, epochs: int):
        X = [item[0] for item in dataset]
        Y = [[item[1]] if type(item[1])!= list else item[1] for item in dataset]
        cost_history = []
        for ep in range(epochs):
            list_a = []
            for s_i in range(len(X)):
                a = self.forward(X[s_i])
                self.backward(self.loss_function_derivative(a, Y[s_i]), learning_rate)
                list_a.append(a)
                
            cost_history.append(self.loss_function(list_a,Y))
            
            if ep % (epochs // 10) == 0:
                print(f"Iteration {ep: 5d}, Cost = {cost_history[ep]:.6f}")
        
                
                
                