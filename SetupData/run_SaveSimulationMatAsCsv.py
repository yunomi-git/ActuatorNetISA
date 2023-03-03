import definitions
from SetupData.ModelDataStructure import ModelDataStructure

from SetupData import SimulationExportToCsv as Export
import SetupData.SimExportNames as SimNames
from Parameters.ModelV1Parameters import _modelV1Structure

def performExport(matFileName : str, modelStructure : ModelDataStructure):
    matlabData = Export.MatlabData(matFileName + ".mat", SimNames.parentRegistryNames)
    input, output = Export.exportSimulationMatToDataframe(actuatorNames=SimNames.actuatorNames,
                                                          numPastStates=modelStructure.numPastStates,
                                                          inputStates=modelStructure.inputStateList,
                                                          outputStates=modelStructure.outputStateList,
                                                          matData=matlabData)

    csvSaveName = modelStructure.getDatasetCsvSaveName(matFileName)
    path = definitions.createPathToCsvDataFile(csvSaveName, isInputs=True)
    print("Exporting Input to csv: " + path)
    input.to_csv(path_or_buf=path, index=False)
    path = definitions.createPathToCsvDataFile(csvSaveName, isInputs=False)
    print("Exporting Output to csv: " + path)
    output.to_csv(path_or_buf=path, index=False)
    print("Finished")

if __name__ == '__main__':
    dataName = "Flatgroundwalking_20220802"

    modelStructure = _modelV1Structure
    saveNameSuffix = modelStructure.getName()

    performExport(matFileName=dataName,
                  modelStructure=modelStructure)

