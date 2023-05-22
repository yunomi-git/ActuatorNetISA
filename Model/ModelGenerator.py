import Model.FullyConnectedNN as fcnn
from enum import Enum

class ModelType(Enum):
    FULLY_CONNECTED = 1
    RECURRENT = 2
    CONVOLUTIONAL = 3

def createNeuralNetwork(type : ModelType, config, dataset, device):
    if type == ModelType.FULLY_CONNECTED:
        return _createFullyConnectedNN(config, dataset, device)
    return None

def _createFullyConnectedNN(model_config, dataset, device):
    modelNetStructure = fcnn.createModelNetStructureFromDataset(numLayers=model_config["num_hidden_layers"],
                                                                sizeLayers=model_config["hidden_layer_size"],
                                                                activationType="",
                                                                dataset=dataset)
    return fcnn.FullyConnectedNeuralNet(mns=modelNetStructure).to(device)