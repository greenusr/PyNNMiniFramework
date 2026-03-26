from .LayerInterface import LayerInterface
import random

class Dense(LayerInterface):
    
    def __init__(self, input_size: int, output_size: int, activation_name: str, lambda_: float):
        super().__init__(input_size, output_size, activation_name, lambda_)
        
        self.Weight = [[random.uniform(-1,1) for _ in range(input_size)] for _ in range(output_size)]
        self.Bias = [0 for _ in range(output_size)]
        
    def forward(self, a: list) -> list:
        #Z = W*A + B
        self.input_data = a
        self.z = [sum(self.Weight[r_i][c_i] * a[c_i] for c_i in range(self.input_size)) + self.Bias[r_i] for r_i in range(self.output_size)]
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, da: list, learning_rate: float) -> list:
        #dl/dz = dl/da * da/dz
        s = self.activation_derivative(self.z)
        dz = [sum(da[c_i] * s[c_i][r_i] for c_i in range(len(s))) for r_i in range(self.output_size)]
        dw = [[self.input_data[c_i] * dz[r_i] for c_i in range(self.input_size)] for r_i in range(self.output_size)]
        db = dz
        
        for r_i in range(self.output_size):
            for c_i in range(self.input_size):
                self.Weight[r_i][c_i] -= learning_rate*(dw[r_i][c_i] + self.lambda_*self.Weight[r_i][c_i])
            self.Bias[r_i] -= learning_rate*db[r_i]
        
        da_prev = [sum(self.Weight[r_i][c_i] * dz[r_i] for r_i in range(self.output_size)) for c_i in range(self.input_size)]
        return da_prev