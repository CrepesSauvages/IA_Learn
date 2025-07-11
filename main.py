import torch

class Layers: 
    def __init__(self, input_size, output_size, activation=torch.relu):
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        self.bias = torch.zeros(output_size, requires_grad=True)
        self.activation = activation
    
    def forward(self, x):
        
        z = x @ self.weight.T + self.bias # x est la matrice des données d'entrée
        return self.activation(z)

    def parameters(self): 
        return [self.weight, self.bias]

