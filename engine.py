import numpy as np
import matplotlib.pyplot as plt
import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        # Initialize the object
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        # Method for adding two Value objects or a value object with a non-Value object
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        # Method for backward pass on a plus operation, while accumulating gradients
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        # Method for multiplying two Value objects or a value object with a non-Value object
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        # Method for backward pass on a muliplication operation, while accumulating gradients
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f"^{other}")
        def _backward():
            self.grad += (other * (self.data)**(other - 1))*out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.e**self.data, (self, ), f"exp")
        def _backward():
            self.grad += (out.data*out.grad)
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(((math.e**(2*self.data) - 1)/(math.e**(2*self.data) + 1)), (self, ), _op='tanh')
        def _backward():
            self.grad += (1-out.data**2)*out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    
    def backward(self):
        # topological order of all children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        print(topo)
        for v in reversed(topo):
            v._backward() 

    def __neg__(self):
        return self*(-1.0)
    
    def __radd__(self, other):
        return self+other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self*other
    
    def __truediv__(self, other):
        return self*(other**-1)
    
    def __rtruediv__(self, other):
        return other*(self**-1)