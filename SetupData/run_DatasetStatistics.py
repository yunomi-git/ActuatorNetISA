import pandas as pd
import numpy as np
from SetupData.Dataset import DatasetFromCsv
from Parameters.ModelV1Parameters import v1ModelDataSummary
from Parameters.ModelV2Parameters import v2ModelDataSummary

# Given some dataframes, prints mean and stdev

import timeit

def getStatisticsOfDataset(csvDataName : str, numPastStates=1):
    dataset = DatasetFromCsv(csvDataName)
    inputs = dataset.x[:, ::numPastStates]
    outputs = dataset.y

    print("Inputs: ")
    print(dataset.xHeaders[::numPastStates])
    printStatisticsOfNumpyData(inputs)

    print("Outputs: ")
    print(dataset.yHeaders)
    printStatisticsOfNumpyData(outputs)

def printStatisticsOfNumpyData(numpyData : np.ndarray):
    means = np.mean(numpyData, axis=0)
    sdvs = np.std(numpyData, axis=0)
    print("means: ")
    print(means)
    print("standard deviations: ")
    print(sdvs)

if __name__ == "__main__":
    matDataName = "Flatgroundwalking_20220802"
    csvDataName = v2ModelDataSummary.modelDataStructure.getDatasetCsvSaveName(matDataName)

    start_time = timeit.default_timer()
    getStatisticsOfDataset(csvDataName, v1ModelDataSummary.modelDataStructure.numPastStates)
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")


    # for column in dataframe.columns:
    #     name = column.name
    #     mean = np.mean(column)
    #     std = np.std(column)