import torch
from SetupData.Dataset import DatasetFromDataframe
from SetupData.DataStateDescription import DataStateList
import SetupData.Parameters.SimExportAndActuatorNames as SimNames
from SetupData.ModelDataStructure import ModelDataStructure
from SetupData import SimulationExportToCsv as Export


class DatasetSummary:
    def __init__(self, data_config):
        self.num_past_history = data_config["num_past_history"]
        self.state_description_name = data_config["state_description_name"]
        if data_config["actuator_name"] == "all":
            self.actuator_names = SimNames.actuatorNames
        else:
            self.actuator_names = [data_config["actuator_name"]]
        self.mat_file_names = data_config["mat_file_name"]

        self.save_name = ("s" + self.state_description_name + "_" +
                          "p" + str(self.num_past_history)  + "_" +
                          data_config["actuator_name"])

    def getSaveName(self):
        return self.save_name


def generateDataset(dataset_summary : DatasetSummary) -> DatasetFromDataframe:
    # Generate the data structure
    state_description = DataStateList.getStateDescription(dataset_summary.state_description_name)
    data_structure = ModelDataStructure(state_description.input_states,
                                        state_description.output_states,
                                        numPastStates=dataset_summary.num_past_history,
                                        name="")

    # Generate file from modelDataStructure, actuator_names, mat file names
    matFileName = dataset_summary.mat_file_names # TODO multiple file names
    matlabData = Export.MatlabData(matFileName + ".mat", SimNames.parentRegistryNames)
    input_data, output_data = Export.exportSimulationMatToDataframe(actuatorNames=dataset_summary.actuator_names,
                                                          numPastStates=data_structure.numPastStates,
                                                          inputStates=data_structure.inputStateList,
                                                          outputStates=data_structure.outputStateList,
                                                          matData=matlabData)

    return DatasetFromDataframe(input_data, output_data)


# def prepare_data(data : DatasetFromDataframe, batch_size):
#     train_data, val_data, test_data = data.get_splits()
#     train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
#     val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
#     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
#     return train_dataloader, val_dataloader, test_dataloader

def prepare_data(data: DatasetFromDataframe, batch_size):
    train_data, val_data, test_data = data.get_splits_semi_random()
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader