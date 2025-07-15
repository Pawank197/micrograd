from engine import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    
class Neurons(Module):
    """
    This class represents a simple neuron with weights and a forward pass.
    A neuron takes multiple inputs, each associated with a weight, and computes a weighted sum.
    """
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
    
    def __call__(self, x, act_fn='tanh'):
        # Forward pass through the neuron: act_fn(weights * inputs + bias)
        act = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = act.tanh() if act_fn == 'tanh' else act.relu()
        return out
    
    def parameters(self):
        # return all weights and bias of the neuron
        return self.weights + [self.bias]
    
class Layer(Module):
    """
    This class represents a layer of neurons in a neural network.
    """
    def __init__(self, nin, nout):
        # create nout neurons, each with nin inputs
        self.neurons = [Neurons(nin) for _ in range(nout)]  
    
    def __call__(self, x):
        # forward pass through each neuron
        outs = [n(x) for n in self.neurons] 
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # return all parameters of all neurons in the layer
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP(Module):
    """
    This class represents a Multi-Layer Perceptron (MLP) neural network.
    Its consructor initializes the neural network with a specified number of input and output neurons,
    and a specified number of hidden layers, each with a specified number of neurons.
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        # forward pass through each layer
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]