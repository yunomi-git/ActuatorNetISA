import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import argparse

from threepotato.three_potato_dataset import RelativePotatoDataset
from threepotato.three_potato_network import ThreePotatoNetwork
from definitions import ROOT_DIR
from threepotato.utils import extract_model_info

from Visualization.Visualize import plot_predictions


def test_model_on_dataset(model_path, dataset_path):
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
    state_dict = torch.load(model_path)
    [num_hidden_layers, hidden_layer_size] = extract_model_info(state_dict)
    model = ThreePotatoNetwork(num_hidden_layers=num_hidden_layers, hidden_layer_size=hidden_layer_size)
    model.load_state_dict(torch.load(model_path))
    print(model)

    data = RelativePotatoDataset(dataset_path)
    inputs = data.x
    actuals = data.y

    predictions = []
    model.eval()
    for row in inputs:
        pred = model(torch.from_numpy(row))
        predictions.append(pred.detach().numpy())
    predictions = np.array(predictions)
    plot_predictions(predictions, actuals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a candidate .pth model on a dataset.")
    parser.add_argument("model_path", nargs=1, help="Path of the .pth file to test")
    parser.add_argument("dataset_path", nargs=1, help="Path of the dataset .csv to be tested")
    args = parser.parse_args()

    test_model_on_dataset(args.model_path[0], args.dataset_path[0])
