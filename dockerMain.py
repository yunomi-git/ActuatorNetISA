import wandb
import argparse
import os
import definitions

from Train.TrainNetwork import model_pipleline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a run of actuator net training.")
    # parser.add_argument("wandb_path", nargs=1, help="Path to the w and b save file")
    parser.add_argument("config_path", nargs=1, help="Path to the training config yaml file")

    args = parser.parse_args()

    config_path = args.config_path[0]
    wandb_path = definitions.WANDB_DIR

    wandb.login(key="97f25c254bd8f91f2c6a99df7b0c7d833a586e81")
    model_pipleline(config=config_path, wandb_path=wandb_path)