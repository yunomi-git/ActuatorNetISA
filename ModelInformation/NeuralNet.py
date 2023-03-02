import torch
import ModelInformation.ModelDataStructure as ModelDataStructure
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
        # Final layer must be appended outside the loop as it doesn't have an activation function after
        self.forwardModel.append(torch.nn.Linear(mns.hiddenLayerSize, mns.outputLayerSize))

    def forward(self, x):
        y = self.forwardModel(x)
        return y