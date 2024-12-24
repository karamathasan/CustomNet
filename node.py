import math
import numpy as np
import random

import activation
from trainable import Weight, Bias

class Node():
    pass

class Neuron(Node):
    def __init__(self, numConnections = None, activationFunc = None):
        self.weights = Weight(numConnections)
        self.bias = Bias()

        self.input = None
        self.activation = 0

        if (activationFunc != None):
            self.activationFunction = activationFunc
        else:
            self.activationFunction = activation.Relu()
    
    def reset(self, numConnections, activationFunc):
        self.weights.reset(numConnections)
        self.bias.reset()

        self.input = None
        self.activation = 0

        if (activationFunc != None):
            self.activationFunction = activationFunc
        else:
            self.activationFunction = activation.Relu()

    def evaluate(self, input: np.ndarray):
        assert (input.shape == self.weights.shape())

        self.input = input
        weightedSum = (self.weights.dot(input)) + self.bias

        activation = self.activationFunction.evaluate(weightedSum)
        self.activation = activation
        return activation
        
    def setActivation(self, activationFunc: activation.ActivationFunction):
        self.activationFunction = activationFunc
        
    def activationFunctionDerivative(self):
        '''
        may break if there has been no function calculation before derivative calculation
        '''
        return self.activationFunction.evaluateDerivative()
    
    def equals(self, other):
        result = True
        if (len(self.weights) != len(other.weights)):
            return False
        for i in range(len(self.weights)):
            result = result and (self.weights[i] == other.weights[i])
        result = result and (self.bias == other.bias) and (self.activationFunction == other.activationFunction)
        return result
    
    def __str__(self):
        return f"weights: {self.weights} \nbias: {self.bias} \nnum connections: {self.numConnections}" 