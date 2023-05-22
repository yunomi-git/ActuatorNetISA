from SetupData.ModelDataStructure import ModelDataStructure
from SetupData.Dataset import DatasetFromDataframe
import torch

class FCModelNetStructure():
    def __init__(self,
                 numLayers : int,
                 sizeLayers : int,
                 inputSize : int,
                 outputSize : int,
                 activationType):
        self.numLayers = numLayers
        self.hiddenLayerSize = sizeLayers
        self.activationType = activationType
        self.inputLayerSize = inputSize
        self.outputLayerSize = outputSize


    def getLayersDims(self):
        layerDims = []
        layerDims.append(self.inputLayerSize)
        for i in range(self.numLayers):
            layerDims.append(self.hiddenLayerSize)
        layerDims.append(self.outputLayerSize)

        return layerDims

class FullyConnectedNeuralNet(torch.nn.Module):
    def __init__(self, mns : FCModelNetStructure):
        super().__init__()
        self.forwardModel = torch.nn.Sequential()

        self.forwardModel.append(torch.nn.Linear(mns.inputLayerSize, mns.hiddenLayerSize))
        self.forwardModel.append(torch.nn.ReLU())
        for i in range(mns.numLayers):
            self.forwardModel.append(torch.nn.Linear(mns.hiddenLayerSize, mns.hiddenLayerSize))
            self.forwardModel.append(torch.nn.ReLU())
        self.forwardModel.append(torch.nn.Linear(mns.hiddenLayerSize, mns.outputLayerSize))

    def forward(self, x):
        y = self.forwardModel(x)
        return y

def createModelNetStructureFromDataStructure(numLayers : int,
                                             sizeLayers : int,
                                             activationType,
                                             modelDataStructure : ModelDataStructure):
    inputLayerSize = modelDataStructure.getInputDimensions()
    outputLayerSize = modelDataStructure.getOutputDimensions()
    return FCModelNetStructure(numLayers=numLayers,
                               sizeLayers=sizeLayers,
                               activationType=activationType,
                               inputSize=inputLayerSize,
                               outputSize=outputLayerSize)

def createModelNetStructureFromDataset(numLayers : int,
                                     sizeLayers : int,
                                     activationType,
                                     dataset : DatasetFromDataframe):
    inputLayerSize = dataset.getInputDimensions()
    outputLayerSize = dataset.getOutputDimensions()
    return FCModelNetStructure(numLayers=numLayers,
                               sizeLayers=sizeLayers,
                               activationType=activationType,
                               inputSize=inputLayerSize,
                               outputSize=outputLayerSize)


