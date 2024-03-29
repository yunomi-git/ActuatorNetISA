# Takes a Mat file from a nadia simulation and turns it into a dataset saved as a Csv
import numpy as np
from typing import Tuple


import SetupData.Parameters.DataNames as dn
import SetupData.Parameters.SimExportNames as sn
import pandas as pd
from enum import Enum
import definitions
import scipy.io
import os



class MatlabData:
    def __init__(self, fileName, parentRegistryList):
        self.matData = self.extractMatDataAsSimplifiedDict(fileName, parentRegistryList)

    def extractMatDataAsSimplifiedDict(self, fileName, parentRegistryList) -> dict:
        path = os.path.join(definitions.DATA_DIR, fileName)
        originalMat = scipy.io.loadmat(path, simplify_cells=True)
        simplifiedMat = originalMat
        for registry in parentRegistryList:
            simplifiedMat = simplifiedMat[registry]

        return simplifiedMat

    def extractColumnFromMatDataByName(self, actuatorName: str, stateName: str) -> np.ndarray:
        simVariableName = self.constructSimulationVariableName(actuatorName, stateName)
        array = self.matData[actuatorName][simVariableName]
        return array

    def constructSimulationVariableName(self, matActuatorName: str, stateName: str):
        return matActuatorName + "_" + stateName

    def getNumDataPoints(self):
        randomStateKey = list(self.matData.keys())[0]
        randomState = self.matData[randomStateKey]
        randomKey = list(randomState.keys())[0]
        return randomState[randomKey].size


def exportSimulationMatToDataframe(actuatorNames : list, numPastStates, inputStates : list, outputStates : list, matData : MatlabData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # actuatorNames is a list of actuators from which to extract data
    # numPastStates is how many past states to augment inputNames with
    # inputStates and outputStates are lists of StateNames and StateDifferenceNames

    # Construct data points
    inputDataFrame = {}
    outputDataFrame = {}
    # [current, current - numPastStates] => current + 1
    numDataForEachActuator = matData.getNumDataPoints() - numPastStates

    # for each state, collect d pieces of data for each actuator
    for stateEnum in inputStates:
        for actuatorName in actuatorNames:
            # Extract the corresponding data from mat
            dataColumn = extractColumnFromMatDataByState(matData, actuatorName, stateEnum)

            # Save the data in corresponding input vector form
            for p in range(numPastStates):
                newStateName = stateEnum.value + "_m" + str(p)
                startingIndex = numPastStates - p - 1
                appendDataToDictKey(inputDataFrame, newStateName, dataColumn[startingIndex: numDataForEachActuator + startingIndex])

    for stateEnum in outputStates:
        for actuatorName in actuatorNames:
            # Extract the corresponding data from mat
            dataColumn = extractColumnFromMatDataByState(matData, actuatorName, stateEnum)

            # Save the data in corresponding input vector form
            newStateName = stateEnum.value + "_next"
            appendDataToDictKey(outputDataFrame, newStateName, dataColumn[numPastStates: numDataForEachActuator + numPastStates])

    return (pd.DataFrame(inputDataFrame), pd.DataFrame(outputDataFrame))


def extractColumnFromMatDataByState(matData : MatlabData, actuatorName: str, stateEnum: Enum) -> np.ndarray:
    if isinstance(stateEnum, dn.StateDifferenceNames):
        measuredEnum, desiredEnum = dn.stateDifferenceMap[stateEnum]
        dataColumn = np.subtract(matData.extractColumnFromMatDataByName(actuatorName, sn.stateNameToSimExportMap[measuredEnum]),
                                 matData.extractColumnFromMatDataByName(actuatorName, sn.stateNameToSimExportMap[desiredEnum]))
    else:
        simName = sn.stateNameToSimExportMap[stateEnum]
        dataColumn = matData.extractColumnFromMatDataByName(actuatorName, simName)

    return dataColumn

def appendDataToDictKey(dictionary : dict, stateName : str, data : np.ndarray):
    if stateName not in dictionary.keys():
        dictionary[stateName] = data
    else:
        dictionary[stateName] = np.append(dictionary[stateName], data)





