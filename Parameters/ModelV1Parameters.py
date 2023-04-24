from SetupData.ModelDataStructure import ModelDataStructure
from ModelInformation.DataNames import StateNames, StateDifferenceNames
from SetupData.ModelDataSummary import ModelDataSummary
from SetupData.DataStatistics import DataStatistics


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
_v1NumPastStates = 3

_modelV1Structure = ModelDataStructure(inputStateList=_v1InputList,
                                       outputStateList=_v1OutputList,
                                       numPastStates=_v1NumPastStates,
                                       name="v1")

_inputStateStatistics = DataStatistics(meansForSingleState=[5.71534395e-01,
                                                            1.50707935e-03,
                                                            1.16655266e+02,
                                                            3.67216580e-02,
                                                            -2.30448008e-01,
                                                            -8.21203709e-01,
                                                            1.18966682e+02,
                                                            -3.17433402e-02,
                                                            1.11009382e-01],
                                       stdsForSingleState=[1.6845012,
                                                          12.766252,
                                                          1019.35815,
                                                          2.8408113,
                                                          1.0341755,
                                                          12.71794,
                                                          1018.627,
                                                          2.817844,
                                                          1.3190902],
                                       numPastStates=_v1NumPastStates)
_outputStateStatistics = DataStatistics(meansForSingleState=[5.7153428e-01,
                                                             1.5062783e-03,
                                                             1.1665521e+02,
                                                             3.6718786e-02,
                                                             -2.3045021e-01],
                                        stdsForSingleState=[1.684502,
                                                           12.766252,
                                                           1019.3581,
                                                           2.8408117,
                                                           1.0341768])

_datasetMatNames = ["Flatgroundwalking_20220802"]

v1ModelDataSummary = ModelDataSummary(modelDataStructure=_modelV1Structure,
                                      inputDataStatistics=_inputStateStatistics,
                                      outputDataStatistics=_outputStateStatistics,
                                      datasetMatNames=_datasetMatNames)