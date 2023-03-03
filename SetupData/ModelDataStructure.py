from ModelInformation.DataNames import StateNames, StateDifferenceNames

class ModelDataStructure:
    # inputStateList and outputStateList are arrays of StateNames or StateDifferenceNames
    def __init__(self,
                 inputStateList : list,
                 outputStateList : list,
                 numPastStates : int,
                 name : str): # Name (ie version) of this specific data structure
        self.inputStateList = inputStateList
        self.outputStateList = outputStateList
        self.numPastStates = numPastStates
        self.name = name


        assert numPastStates >= 1, \
            "num past states must be >= 1"
        assert (isinstance(inputStateList[0], StateNames) or isinstance(inputStateList[0], StateDifferenceNames)), \
            "inputStateList has wrong type"
        assert (isinstance(outputStateList[0], StateNames) or isinstance(outputStateList[0], StateDifferenceNames)), \
            "outputStateList has wrong type"

    def getInputDimensions(self):
        return len(self.inputStateList) * self.numPastStates

    def getOutputDimensions(self):
        return len(self.outputStateList)

    def getName(self):
        return self.name + "p" + str(self.numPastStates)

    def getDatasetCsvSaveName(self, datasetBaseName):
        return datasetBaseName + "_" + self.getName()