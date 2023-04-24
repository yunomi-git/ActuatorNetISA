import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # this will give us the project root dir

# This is for windows running
# DATA_DIR = os.path.join(ROOT_DIR, "..", "TrainingData")
# WANDB_DIR = os.path.join(ROOT_DIR, "..")
# This is for gpu server running
DATA_DIR = os.path.join(ROOT_DIR, "data", "TrainingData")
WANDB_DIR = os.path.join(ROOT_DIR, "data")


def createPathToCsvDataFile(saveName, isInputs : bool):
    extension = "_outputs.csv"
    if isInputs:
        extension = "_inputs.csv"

    return os.path.join(DATA_DIR, saveName + extension)