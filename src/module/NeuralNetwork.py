from turtle import forward
import torch
from main import Layers

class NeuralNetwork:
    def __init__(self, layer_sizes, activation=torch.relu):
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            acts = activation if i < len(layer_sizes) - 2 else lambda x: x
            self.layers.append(Layers(layer_sizes[i], layer_sizes[i+1], activation=acts))


    def forward(self, x):
        # x : tensor taille
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def param(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
