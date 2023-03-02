import definitions
import os
from ModelInformation.ModelDataStructure import ModelDataStructure

from SetupData import SimulationExportToCsv as Export
import SetupData.SimExportNames as SimNames
from Parameters.ModelV1Parameters import modelV1Structure

def performExport(dataFileName : str, saveNameSuffix : str, modelStructure : ModelDataStructure):
    matlabData = Export.MatlabData(dataFileName + saveNameSuffix + ".mat", SimNames.parentRegistryNames)
    input, output = Export.exportSimulationMatToDataframe(actuatorNames=SimNames.actuatorNames,
                                                          numPastStates=modelStructure.numPastStates,
                                                          inputStates=modelStructure.inputStateList,
                                                          outputStates=modelStructure.outputStateList,
                                                          matData=matlabData)

    path = definitions.createPathToCsvDataFile(dataFileName, isInputs=True)
    print("Exporting Input to csv: " + path)
    input.to_csv(path_or_buf=path, index=False)
    path = definitions.createPathToCsvDataFile(dataFileName, isInputs=False)
    print("Exporting Output to csv: " + path)
    output.to_csv(path_or_buf=path, index=False)
    print("Finished")

if __name__ == '__main__':
    dataName = "Flatgroundwalking_20220802"

    modelStructure = modelV1Structure
    saveNameSuffix = modelStructure.getName()

    performExport(dataFileName=dataName,
                  saveNameSuffix=saveNameSuffix,
                  modelStructure=modelStructure)

