import wandb
import argparse

from Train.TrainNetwork import model_pipleline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a run of three potato residual training.")
    parser.add_argument("config_path", nargs=1, help="Path to the training config yaml file")
    args = parser.parse_args()

    wandb.login()
    model_pipleline(config=args.config_path[0])