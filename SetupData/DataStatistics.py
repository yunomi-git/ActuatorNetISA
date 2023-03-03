# For a given set of states & past states, holds the means and standard deviation

import torch
import numpy as np
class DataStatistics:
    def __init__(self, meansForSingleState : list, stdsForSingleState : list, numPastStates=1):
        # if single state is [1, 2, 3] with past states 2, the full state is [1 1 2 2 3 3]
        self.means = np.array(meansForSingleState).repeat(numPastStates).astype('float32')
        self.stds = np.array(stdsForSingleState).repeat(numPastStates).astype('float32') # TODO is there a better place ot put this

    def getNormalizationScaling(self):
        return self.stds.__pow__(-1.0)

    def getUnNormalizationScaling(self):
        return self.stds

    # def applyNormalization(self, vector : torch.Tensor) -> torch.Tensor:
    #     scale = self.stds.pow(-1)
    #     # return torch.mul(torch.subtract(vector, self.means), scale)
    #     return torch.mul(vector, scale)
    #
    # def applyUnNormalization(self, vector : torch.Tensor) -> torch.Tensor:
    #     # return torch.mul(torch.add(vector, self.means), self.stds)
    #     return torch.mul(vector, self.stds)