import math
import numpy as np
import matplotlib.pyplot as plt
import random
from engine import Value


# 构建MLP
class Neuron():

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]
class Layer():

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP():

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

def train(net, epoches, learning_rate):
    for epoch in range(epoches):
        # forward
        ypred = [net(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        # zero_grad
        for p in net.parameters():
            p.grad = 0.0
        # backward
        loss.backward()
        # update
        for p in net.parameters():
            p.data -= learning_rate*p.grad
        print(f'Epoch: {epoch}, Loss:{loss.data}')

net = MLP(3, [4, 4, 1])
train(net, 50, 0.01)