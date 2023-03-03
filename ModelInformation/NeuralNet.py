import torch
import ModelInformation.ModelNetStucture as ModelNetStructure

class NeuralNet(torch.nn.Module):
    def __init__(self, mns : ModelNetStructure):
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