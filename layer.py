import numpy as np
import node as n

from trainable import Trainable

'''
the layer contains neurons. each neuron has an array/vector of weights associated with its output, another array/ vector of biases and an activation function
the length of these vectors should be equal to the size of the next layer
'''
class Layer():
    pass
    def evaluate():
        pass
    def reset():
        pass

class TrainableLayer(Layer):
    def __init__(self):
        self.trainable: list[Trainable]

    def train(self):
        pass

    def dOutdIn(self):
        """
        returns the derivative of this layers input with respect to its output
        """
        pass

class Dense(TrainableLayer):
    # the direction of connections is backward 
    def __init__(self, size, connections = None, activationFunc = None):
        self.neurons = []
        self.connections = connections
        self.size = size
        if connections:
            for i in range(size):
                # generate 'size' many neurons with random weights and biases that output to 'connections' many inputs in the preceding layer 
                self.neurons.append(n.Neuron(connections, activationFunc))
    
    def reset(self, connections, activationFunc = None):
        assert connections
        self.neurons = []
        self.connections = connections
        for i in range(self.size):
            self.neurons.append(n.Neuron(connections, activationFunc))

    def evaluate(self, input: np.array):
        assert(input.shape[0] == self.connections)
        output = np.zeros(self.size)
        for i in range(self.size):
            output[i] = self.neurons[i].evaluate(input)
        return output
    
    def equals(self, other):
        result = True
        if (self.size != len(other.neurons)):
            return False
        for i in range(self.size):
            result = result and (self.neurons[i].equals(other.neurons[i]))
        result = result and (self.connections == other.connections)
        return result
    
    def getNeurons(self):
        return self.neurons

    def getLayerActivation(self):
        layerActivation = np.array([])
        for n in self.neurons:
            layerActivation = np.append(layerActivation, n.activation)
        return layerActivation
    
    def getLayerActivationDerivative(self):
        activationDerivative = np.array([])
        for neuron in self.neurons:
            activationDerivative = np.append(activationDerivative, neuron.activationDerivative())
        return activationDerivative
    

class Dropout(Layer):
    pass

class BatchNormalization(TrainableLayer):
    pass

class Convulational2D(TrainableLayer):
    pass

class Convulational3D(TrainableLayer):
    pass