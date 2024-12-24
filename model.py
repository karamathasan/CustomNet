import numpy as np
import pandas as pd

from log import Log, EpochLog
import metrics
import loss 
import activation
import optimizer as opt 

from layer import TrainableLayer, Layer
from data_helper import shuffle
from backprop_helper import backpropagate

class Model():
    def __init__(self):
        self.layers: list[Layer]
    pass
    def compile():
        pass
    def predict():
        pass
    def train():
        pass
    def add():
        pass

class Sequential():
    '''
    create a neural network model with sequential layers

    Parameters:

        activationFunc: an activation function that newly added layers by default will use. ReLU by default
        lossFunc: a loss function used for training. Squared error by default
        optimizer: the optimizer used for training. Stochastic Gradient Descent by default 
            >>>(it is not quite SGD, it applies gradient descent to whatever batch size is present)
        gradient_clipping_magnitude: the magnitude of a gradient before it gets clipped. None by default, meaning no clip is done
        normalize_weights: allow the normalization of weights through the use of turning each weight into a multiplication of a vector and a scalar. False by default
    '''
    def __init__(
            self, *layers: Layer, input_size: int, output_size: int,  activationFunc = activation.Relu(), 
            lossFunc = loss.SquaredError(), optimizer = opt.SGD(0.001), 
            gradient_clipping_magnitude = None, normalize_weights = False, metrics = metrics.SquaredError()
        ):
        self.layers = [*layers]
        self.input_size = input_size
        self.output_size = output_size
        
        assert isinstance(activationFunc, activation.ActivationFunction), "ACTIVATION FUNCTION PARAMETER IS NOT OF TYPE ACTIVATIONFUNCTION"
        assert isinstance(lossFunc, loss.LossFunction), "LOSS FUNCTION PARAMETER IS NOT OF TYPE LOSSFUNCTION"
        assert isinstance(optimizer, opt.Optimizer), "OPTIMIZER PARAMETER IS NOT OF TYPE OPTIMIZER"
 
        self.defaultActivation = activationFunc # a default activation is made so that newly added activation functions will be set to this activationFunction
        self.lossFunc = lossFunc
        self.optimizer = optimizer

        self.gradient_clipping_magnitude = gradient_clipping_magnitude
        self.normalize_weights = normalize_weights
        self.metrics = metrics
    
    def compile(self):
        self.layers[0].reset(self.input_size, self.defaultActivation)
        for i in range(len(self.layers)-1):
            self.layers[i+1].reset(self.layers[i].size, self.defaultActivation)
        assert self.output_size == self.layers[-1].size

    def predict(self, input):
        output = input
        for i in range(len(self.layers)):
            currentLayer = self.layers[i]
            output = currentLayer.evaluate(output)
        return output

    def train(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame, validation_predictor: pd.DataFrame, validation_effector: pd.DataFrame, batch_size, epochs = 1):
        '''
        train the model 

        Args:
            predictor_data: dataframe with the input data to predict an output
            effector_data: dataframe with the corresponding true outputs to the input
            batch_size: int or string referring to the number of datapoints used to accumulate error for a backward pass
                >>>accepted strings: "full-batch", "stochastic"
        '''
        assert (predictor_data.shape[0] == effector_data.shape[0]) 
        iterative_log = Log(self)
        epoch_log = EpochLog(self)

        for j in range(epochs): 
            print(f"epoch {j + 1} begin")
            if (type(batch_size) == int):
                assert (batch_size > 1), (f"INVALID INPUT FOR BATCH_SIZE, {batch_size}")
                residuals = 0
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()   
                    true = effector_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    residuals += self.calculateResidualsArray(true,prediction)

                    iterative_log.update(true,prediction)
                    if (i == 0):
                        epoch_log.update(true,prediction)

                        validation_true = validation_effector.iloc[i].to_numpy()
                        validation_prediction = self.predict(validation_predictor.iloc[i].to_numpy())
                        epoch_log.updateValidation(validation_true, validation_prediction)

                    if((i % batch_size == 0 or i == len(predictor_data)) and i != 0):
                        backpropagate(self,residuals)
                        residuals = 0

            elif batch_size == "full-batch":
                residuals = 0
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    true = effector_data.iloc[i].to_numpy()
                    residuals += self.calculateResidualsArray(true, prediction)
                    
                    iterative_log.update(true,prediction)
                    if (i == 0):
                        epoch_log.update(true,prediction)

                        validation_true = validation_effector.iloc[i].to_numpy()
                        validation_prediction = self.predict(validation_predictor.iloc[i].to_numpy())
                        epoch_log.updateValidation(validation_true, validation_prediction)
                
                backpropagate(self,residuals)

            elif batch_size == "stochastic":
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    true = effector_data.iloc[i].to_numpy()
                    residuals = self.calculateResidualsArray(true, prediction)
                    print(f"iteration: {j,i}")
                    print(f"    prediction: {prediction}")
                    print(f"    true: {true}")
                    print(f"    residual: {residuals}")

                    iterative_log.update(true,prediction)
                    if (i == 0):
                        epoch_log.update(true,prediction)

                        validation_true = validation_effector.iloc[i].to_numpy()
                        validation_prediction = self.predict(validation_predictor.iloc[i].to_numpy())
                        epoch_log.updateValidation(validation_true, validation_prediction)

                    backpropagate(self,residuals)
            else:
                return ValueError(f"INVALID INPUT FOR BATCH_SIZE, {batch_size}")
        print("training complete!")

        epoch_log.addLossPlot()
        epoch_log.addValidationLossPlot()
        epoch_log.graph()

        epoch_log.addPreformancePlot()
        epoch_log.addValidationPreformancePlot()
        epoch_log.graph()

        iterative_log.addLossPlot()
        iterative_log.graph()

        iterative_log.addPreformancePlot()
        iterative_log.graph()

    def test(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame):
        metric_mean = 0
        for i in range(len(predictor_data)):
            prediction = self.predict(predictor_data.iloc[i].to_numpy())
            truth = effector_data.iloc[i].to_numpy()
            metric_mean += self.metrics.evaluate(truth,prediction)
        metric_mean = metric_mean / len(predictor_data)
        print(f'Mean {self.metrics.getName()}: {metric_mean}')

    def calculateResidualsArray(self,y_true,y_pred):
        return self.lossFunc.evaluateAsArray(y_true,y_pred)
    
    def calculateResidualsSum(self,y_true,y_pred):
        return self.lossFunc.evaluateAsSum(y_true,y_pred)

    def add(self, layer):
        self.layers.append(layer)

    # def modelShape(self):
    #     print(f"input: {self.getInputSize()}")
    #     for layerNum in range(len(self.hiddenLayers)):
    #         print(f"hidden layer {layerNum}: {self.hiddenLayers[layerNum].getSize()}") 
    #     print(f"output: {self.getOutputSize()}") 

    # def neuronCount(self):
    #     count = 0
    #     for layer in self.hiddenLayers:
    #         count += len(layer.getNeurons())
    #     count += len(self.outputLayer.getNeurons())

    # def getNeurons(self):
    #     neurons = []
    #     for hiddenLayer in self.hiddenLayers:
    #         neurons.append(hiddenLayer.getNeurons())
    #     neurons.append(self.outputLayer.getNeurons())
    #     return neurons
    
    def getLayerByIndex(self,index):
        return self.layers[index]
    
    def getLayers(self):
        '''
        returns layers as an array
        '''
        return self.layers
    
    # def getParams(self, printParams = False):
    #     modelParams = []
    #     for j in range(len(self.getLayers())):
    #         modelParams.append([])
    #         for i in range(len(self.getLayers()[j].neurons)):
    #             if (printParams is True):
    #                 print(f"layer: {j}, neuron: {i}")
    #                 print(f"    weights: {self.getLayers()[j].neurons[i].weights}")
    #                 print(f"    bias: {self.getLayers()[j].neurons[i].bias}")
    #             neuronParams = [self.getLayers()[j].neurons[i].weights, self.getLayers()[j].neurons[i].bias]
    #             modelParams[j].append(neuronParams)
    #     return modelParams
                
    # def getParamDifference(self, oldParams, newParams):
    #     modelDiff = []
    #     for i in range(len(oldParams)):
    #         modelDiff.append([])
    #         for j in range(len(oldParams[i])):
    #             layerWeightDiff = newParams[i][j][0] - oldParams[i][j][0]
    #             layerBiasDiff = newParams[i][j][1] - oldParams[i][j][1]
    #             modelDiff[i].append([layerWeightDiff, layerBiasDiff])

    #     for j in range(len(modelDiff)):
    #         for i in range(len(modelDiff[j])):
    #             print(f"layer: {j}, neuron: {i}")
    #             print(f"    weights change: {modelDiff[j][i][0]}")
    #             print(f"    bias change: {modelDiff[j][i][1]}")
    #     return modelDiff 
    
    def getLearningRate(self):
        return self.optimizer.learningRate