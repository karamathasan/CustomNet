import numpy as np
from abc import ABC
class Trainable(ABC):
    def __init__(self):
        self.val = None
        
class Weight(Trainable):
    def __init__(self,size):
        self.size = size

class Bias(Trainable):
    def __init__(self):
        self.val = 0