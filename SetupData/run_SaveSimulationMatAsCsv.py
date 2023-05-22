import definitions
from SetupData.ModelDataStructure import ModelDataStructure

from SetupData import SimulationExportToCsv as Export
import SetupData.Parameters.SimExportNames as SimNames
from Parameters.ModelV2Parameters import _modelV2Structure
import timeit



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

    modelStructure = _modelV2Structure
    saveNameSuffix = modelStructure.getName()

    start_time = timeit.default_timer()
    performExport(matFileName=dataName,
                  modelStructure=modelStructure)
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")

