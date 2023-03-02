from enum import Enum

class StateNames(Enum):
    measuredPosition = "measuredPosition"
    measuredVelocity = "measuredVelocity"
    measuredForce = "measuredForce"
    measuredSpool = "measuredSpool"
    measuredCurrent = "measuredCurrent"

    desiredPosition = "desiredPosition"
    desiredVelocity = "desiredVelocity"
    desiredForce = "desiredForce"
    desiredSpool = "desiredSpool"
    desiredCurrent = "desiredCurrent"

    pressureSupply = "pressureSupply"
    pressurePull = "pressurePull"
    pressurePush = "pressurePush"

class StateDifferenceNames(Enum):
    errorPosition = "errorPosition"
    errorVelocity = "errorVelocity"
    errorForce = "errorForce"
    errorSpool = "errorSpool"
    errorCurrent = "errorCurrent"

stateDifferenceMap = {
    StateDifferenceNames.errorPosition : (StateNames.measuredPosition, StateNames.desiredPosition),
    StateDifferenceNames.errorVelocity : (StateNames.measuredPosition, StateNames.desiredPosition),
    StateDifferenceNames.errorForce : (StateNames.measuredPosition, StateNames.desiredPosition),
    StateDifferenceNames.errorSpool : (StateNames.measuredPosition, StateNames.desiredPosition),
    StateDifferenceNames.errorCurrent : (StateNames.measuredPosition, StateNames.desiredPosition)
}
