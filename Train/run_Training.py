import wandb
import argparse
import os
import definitions

from Train.TrainNetwork import model_pipleline

if __name__ == "__main__": #only run the code if it's in interpreter
    parser = argparse.ArgumentParser(description="Execute a run of actuator net training.")
    # parser.add_argument("wandb_path", nargs=1, help="Path to the w and b save file")
    parser.add_argument("config_path", nargs=1, help="Path to the training config yaml file")
    #?? Where is "config_path" declared or associated??

    args = parser.parse_args()

    config_path = args.config_path[0]

    wandb_path = definitions.WANDB_DIR

    wandb.login(key="9c15968729693673cfd624bada3213715af920d5")
    model_pipleline(config=config_path, wandb_path=wandb_path)