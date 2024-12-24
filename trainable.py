import numpy as np
import random

from abc import ABC
class Trainable(ABC):
    pass
    def __init__(self):
        self.val = 0.0

    def update():
        pass

    def __add__(self,other):
        return self.val + other
    
    def __radd__(self,other):
        return self.val + other
    
    def __mul__(self,other):
        return self.val * other
    
    def __lmul__(self,other):
        return other * self.val
    
    def __truediv__(self,other):
        return self.val / other
    
    def reset(self):
        pass

class Weight(Trainable):
    """
    Create a weight vector for typical dense layer neurons
    """
    def __init__(self,size):
        if size:
            self.size = size
            self.val = np.sqrt(1/size) * self.createRandoms(-1,1,size)
    
    def reset(self, size):
        if size:
            self.size = size
            self.val = np.sqrt(1/size) * self.createRandoms(-1,1,size)

    def createRandoms(self, min, max, quantity):
        randoms = np.array([])
        for _ in range(quantity):
            randoms = np.append(randoms, random.uniform(min,max))
        return randoms
    
    def dot(self, other: np.ndarray):
        return self.val.dot(other)

    def shape(self):
        return self.val.shape

class Bias(Trainable):
    def __init__(self):
        self.val = 0.0

    def reset(self):
        self.val = 0.0
