from ModelInformation.ModelDataStructure import ModelDataStructure
from ModelInformation.DataNames import StateNames, StateDifferenceNames


v1InputList = [StateDifferenceNames.errorPosition,
               StateNames.measuredVelocity,
               StateNames.measuredForce,
               StateNames.measuredSpool,
               StateNames.measuredCurrent,
               StateNames.desiredVelocity,
               StateNames.desiredForce,
               StateNames.desiredSpool,
               StateNames.desiredCurrent]

v1OutputList = [StateDifferenceNames.errorPosition,
                StateNames.measuredVelocity,
                StateNames.measuredForce,
                StateNames.measuredSpool,
                StateNames.measuredCurrent]
v1NumPastStates = 3;

modelV1Structure = ModelDataStructure(inputStateList=v1InputList,
                                      outputStateList=v1OutputList,
                                      numPastStates=v1NumPastStates,
                                      name="v1")