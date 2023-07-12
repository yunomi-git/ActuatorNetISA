from SetupData.Parameters.DataNames import StateNames

parentRegistryNames = ["root", "main", "DRCEstimatorThread", "NadiaEtherCATRealtimeThread", "HardwareMap"]

# actuatorNames = ["rightKneeISA",
#                  "rightHipInnerISA",
#                  "rightHipOuterISA",
#                  "rightAnkleOuterISA",
#                  "rightAnkleInnerISA",
#                  "leftKneeISA",
#                  "leftHipInnerISA",
#                  "leftHipOuterISA",
#                  "leftAnkleOuterISA",
#                  "leftAnkleInnerISA",
#                  "leftSpineISA",
#                  "rightSpineISA"
#                  ]

actuatorNames = ["rightKneeISA",
                 "rightHipInnerISA",
                 "rightHipOuterISA",
                 "rightAnkleOuterISA",
                 "rightAnkleInnerISA",
                 "leftKneeISA",
                 "leftHipInnerISA",
                 "leftHipOuterISA",
                 "leftAnkleOuterISA",
                 "leftAnkleInnerISA",
                 "leftSpineISA",
                 "rightSpineISA"
                 ]

stateNameToSimExportMap = {
    StateNames.measuredPosition : "actualStrokePositionMM",
    StateNames.measuredVelocity : "velocityMMPS",
    StateNames.measuredForce : "temperatureCompensatedForceFromISA",
    StateNames.measuredSpool : "actualSpoolPosition",
    StateNames.measuredCurrent : "measuredCurrent",

    StateNames.desiredPosition : "desiredStrokeLengthMM",
    StateNames.desiredVelocity :"desiredVelocityMMPS",
    StateNames.desiredForce : "desiredForceFromISA",
    StateNames.desiredSpool : "desiredSpoolPositionFromISA",
    StateNames.desiredCurrent : "desiredCurrentFromISA",

    StateNames.pressureSupply : "pressureSupplyLowPass",
    StateNames.pressurePull : "pressurePullChamberLowPass",
    StateNames.pressurePush : "pressurePushChamberLowPass"
}