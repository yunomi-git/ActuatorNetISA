from Parameters.ModelV1Parameters import v1ModelDataSummary

if __name__ == '__main__':
    dataName = "Flatgroundwalking_20220802"
    # datasetFromCsv = DatasetFromCsv(dataName)

    dataset = v1ModelDataSummary.getDatasetForMatName(dataName)

    # modelStructure = modelV1Structure
    # matlabData = Export.MatlabData(dataName + ".mat", SimNames.parentRegistryNames)
    # input, output = Export.exportSimulationMatToDataframe(actuatorNames=SimNames.actuatorNames,
    #                                                       numPastStates=modelStructure.numPastStates,
    #                                                       inputStates=modelStructure.inputStateList,
    #                                                       outputStates=modelStructure.outputStateList,
    #                                                       matData=matlabData)
    # datasetFromDataframe = DatasetFromDataframe(input, output)
    print("")