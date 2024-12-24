from model import Model, Sequential
from layer import TrainableLayer

class Trainer():
    def __init__(self, model: Model):
        self.model = model

    # @staticmethod
    # def backprop(model:Model):

    def backprop(model:Sequential):
        numLayers = len(model.layers)
        # does not assume there are any droupout or batch normalization layers

        """
        the gradient of trainable parameter is dependent on the impact it holds on
        its own layer and the downstream units and layers.
        dLdW = dLdX_k * dX_kdX_j * dX_jdX_i * dX_idW
        each of these terms may arise from expansions of the chain rule
        """

        # for i in range(numLayers):
        #     curr = model.layers[numLayers - i - 1]
        #     if isinstance(curr, TrainableLayer):

