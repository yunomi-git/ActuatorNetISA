from SetupData.Parameters.DataNames import StateDifferenceNames, StateNames

class DataStateDescription():
    def __init__(self, input_states, output_states):
        self.input_states = input_states
        self.output_states = output_states

# --------------------- V1 -------------------------------
_v1InputList = [StateDifferenceNames.errorPosition,
                StateNames.measuredVelocity,
                StateNames.measuredForce,
                StateNames.measuredSpool,
                StateNames.measuredCurrent,
                StateNames.desiredVelocity,
                StateNames.desiredForce,
                StateNames.desiredSpool,
                StateNames.desiredCurrent]

_v1OutputList = [StateDifferenceNames.errorPosition,
                 StateNames.measuredVelocity,
                 StateNames.measuredForce,
                 StateNames.measuredSpool,
                 StateNames.measuredCurrent]

v1DataState = DataStateDescription(_v1InputList, _v1OutputList)

# --------------------- V2 -------------------------------
_v2InputList = [StateNames.measuredVelocity,
                StateNames.measuredForce,
                StateNames.measuredSpool,
                StateNames.desiredVelocity,
                StateNames.desiredForce,
                StateNames.desiredSpool]

_v2OutputList = [StateNames.measuredVelocity,
                 StateNames.measuredForce]

v2DataState = DataStateDescription(_v2InputList, _v2OutputList)

# --------------------- Map -------------------------------
class DataStateList:
    dataStateList = {
        "v1" : v1DataState,
        "v2" : v2DataState
    }

    def getStateDescription(name : str) -> DataStateDescription:
        return DataStateList.dataStateList[name]
