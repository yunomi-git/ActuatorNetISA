from ModelInformation.DataNames import StateNames, StateDifferenceNames

class ModelDataStructure:
    # inputStateList and outputStateList are arrays of StateNames or StateDifferenceNames
    def __init__(self, inputStateList, outputStateList, numPastStates, name):
        self.inputStateList = inputStateList
        self.outputStateList = outputStateList
        self.numPastStates = numPastStates
        self.name = name

        if numPastStates < 1:
            assert "num past states must be >= 1"
        if not (isinstance(inputStateList[0], StateNames) or isinstance(inputStateList[0], StateDifferenceNames)):
            assert "inputStateList has wrong type"
        if not (isinstance(outputStateList[0], StateNames) or isinstance(outputStateList[0], StateDifferenceNames)):
            assert "outputStateList has wrong type"

    def getInputDimensions(self):
        return len(self.inputStateList) * self.numPastStates

    def getOutputDimensions(self):
        return len(self.outputStateList)

    def getName(self):
        return self.name + "p" + str(self.numPastStates)

    # TODO something to interpret the content of a vector. ex x[4] is measuredTorque_past1