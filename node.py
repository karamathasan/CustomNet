import math
import numpy as np
import random

import activation

class Node():
    pass

class Neuron(Node):
    def __init__(self, numConnections = None, activationFunc = None):
        # self.weights = self.createRandoms( -1,1,numConnections)
        if numConnections:
            self.weights = self.initializeWeights(numConnections)
        self.numConnections = numConnections
        # self.bias = random.uniform(-1,1)
        self.bias = 0

        self.input = None
        self.activation = 0

        if (activationFunc != None):
            self.activationFunction = activationFunc
        else:
            self.activationFunction = activation.Relu()
    
    def reset(self, numConnections, activationFunc):
        if numConnections:
            self.weights = self.initializeWeights(numConnections)
        self.numConnections = numConnections
        self.bias = 0

        self.input = None
        self.activation = 0

        if (activationFunc != None):
            self.activationFunction = activationFunc
        else:
            self.activationFunction = activation.Relu()

    def initializeWeights(self, n):
        weights = self.createRandoms(-1,1,n)
        return weights * np.sqrt(1/ n)
    
    def createRandoms(self, min, max, quantity):
        randoms = np.array([])
        for _ in range(quantity):
            randoms = np.append(randoms, random.uniform(min,max))
        return randoms

    def evaluate(self, input: np.array):
        assert(input.shape == self.weights.shape)
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