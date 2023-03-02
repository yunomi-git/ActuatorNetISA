import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # this will give us the project root dir
DATA_DIR = os.path.join(ROOT_DIR, "Data")

def createPathToCsvDataFile(dataName, isInputs : bool):
    extension = "_outputs.csv"
    if isInputs:
        extension = "_inputs.csv"

    return os.path.join(DATA_DIR, dataName + extension)