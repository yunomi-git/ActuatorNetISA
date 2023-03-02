from ModelInformation.ModelDataStructure import ModelDataStructure
from SetupData.Dataset import DatasetFromDataframe
class ModelNetStructure():
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

def createModelNetStructureFromDataStructure(numLayers : int,
                                             sizeLayers : int,
                                             activationType,
                                             modelDataStructure : ModelDataStructure):
    inputLayerSize = modelDataStructure.getInputDimensions()
    outputLayerSize = modelDataStructure.getOutputDimensions()
    return ModelNetStructure(numLayers=numLayers,
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
    return ModelNetStructure(numLayers=numLayers,
                             sizeLayers=sizeLayers,
                             activationType=activationType,
                             inputSize=inputLayerSize,
                             outputSize=outputLayerSize)


