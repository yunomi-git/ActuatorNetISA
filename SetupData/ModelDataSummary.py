# Holds a data structure and references to the datasets it created

from SetupData.ModelDataStructure import ModelDataStructure
from SetupData.Dataset import DatasetFromCsv
from SetupData.DataStatistics import DataStatistics
class ModelDataSummary:
    def __init__(self,
                 modelDataStructure : ModelDataStructure,
                 datasetMatNames: list, # List of names of simulations that created datasets
                 inputDataStatistics : DataStatistics,
                 outputDataStatistics : DataStatistics):

        self.modelDataStructure = modelDataStructure
        self.datasetMatNames = datasetMatNames
        self.inputDataStatistics = inputDataStatistics
        self.outputDataStatistics = outputDataStatistics

    def getDatasetForMatName(self, datasetMatName):
        assert datasetMatName in self.datasetMatNames, "Dataset has not been created for this model"

        datasetCsvName = self.modelDataStructure.getDatasetCsvSaveName(datasetMatName)
        return DatasetFromCsv(datasetCsvName,
                              inputStatistics=self.inputDataStatistics,
                              outputStatistics=self.outputDataStatistics)




