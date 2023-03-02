import pandas as pd
import numpy as np
from SetupData.Dataset import DatasetFromCsv
# Given some dataframes, prints mean and stdev

def getStatisticsOfDataset(datasetName : str):
    dataset = DatasetFromCsv(datasetName)
    inputs = dataset.x
    outputs = dataset.y

    print("Inputs: ")
    print(dataset.xHeaders)
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
    dataName = "Flatgroundwalking_20220802"
    getStatisticsOfDataset(dataName)

    # for column in dataframe.columns:
    #     name = column.name
    #     mean = np.mean(column)
    #     std = np.std(column)