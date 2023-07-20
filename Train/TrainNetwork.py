import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import wandb
from tqdm import tqdm
import os
from datetime import datetime
import timeit
import pylab as pl

import definitions
from Visualization.Visualize import plot_predictions
import Model.ModelGenerator as ModelGenerator
import SetupData.DatasetGenerator as DatasetGenerator

#######
#1. How to expand wandb logged data?
#######

def model_pipleline(config=None, wandb_path=definitions.WANDB_DIR):
    # USE FOR DEBUG PURPOSES ONLY!
    # torch.manual_seed(45)
    # torch.use_deterministic_algorithms(True)

    with wandb.init(project="actuator-net", config=config, dir=wandb_path):  # specifying wandb dir to always be located in project root
        config = wandb.config

        # Start timing
        start_time = timeit.default_timer()

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device count:', torch.cuda.device_count())
        print('Using device:', device)
        print()

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        # Set up the data
        dataset_summary = DatasetGenerator.DatasetSummary(config.data)
        print("Generating Data...")
        generate_start = timeit.default_timer()
        dataset = DatasetGenerator.generateDataset(dataset_summary)
        generate_elapsed = timeit.default_timer() - generate_start
        print("Data Generated. Time taken: " + str(generate_elapsed))
        # Assign Data
        train_dataloader, val_dataloader, test_dataloader = DatasetGenerator.prepare_data(dataset,
                                                                                          config.data["batch_size"])
        # Set up the model
        model = ModelGenerator.createNeuralNetwork(ModelGenerator.ModelType.FULLY_CONNECTED,
                                                   config.model,
                                                   dataset,
                                                   device)
        print(model)

        # Set up the optimization
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training["learning_rate"])

        # Training and validation logic
        wandb.watch(model, criterion, log="all", log_freq=10)

        train_example_count = 0  # number of training examples seen
        val_example_count = 0
        train_batch_count = 0  # number of batches gone through
        val_batch_count = 0
        step = 0  # monotonically increasing step for wandb logging
        print(f"Total Training Epoch" + str(config.training["epochs"]))
        print(f"Learning Rate " + str(config.training["learning_rate"]))

        for epoch in tqdm(range(config.training["epochs"])): #count in range with progression bar tqdm
            # Training
            model.train()
            for i, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # clear the gradients

                yhat = model(inputs)  # compute the model output
                train_loss = criterion(yhat, targets)  # calculate loss

                train_loss.backward()  # credit assignment
                optimizer.step()  # update model weights

                train_example_count += len(inputs)
                train_batch_count += 1
                # Report metrics every 25th batch
                if (train_batch_count + 1) % 25 == 0:
                    wandb.log({"epoch": epoch, "train_loss": train_loss}, step=step)
                #     print(f"Training loss after " + str(train_example_count) + f" examples: {train_loss:.3f}")

            # Validation
            model.eval()  # Evaluation mode
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    yhat = model(inputs)
                    val_loss = criterion(yhat, targets)

                    val_example_count += len(inputs)
                    val_batch_count += 1
                    # Report metrics every 25th batch
                    if (val_batch_count + 1) % 25 == 0:
                        wandb.log({"epoch": epoch, "val_loss": val_loss}, step=step)
                        #print(f"Validation loss after " + str(val_example_count) + f" examples: {val_loss:.3f}")

            step += 1


        # Testing logic
        model.eval()
        with torch.no_grad():
            predictions, actuals = list(), list()
            for i, (inputs, targets) in enumerate(test_dataloader):
                inputs = inputs.to(device)  # only send inputs to GPU, as targets don't need to go through model
                yhat = model(inputs)
                yhat = yhat.cpu()
                yhat = yhat.detach().numpy()
                actual = targets.numpy()
                predictions.append(yhat)
                actuals.append(actual)
            predictions, actuals = np.vstack(predictions), np.vstack(actuals)

            pl.figure(figsize=(14,6))
            pl.plot(actuals[:, 2])  #Velocity
            pl.plot(actuals[:, 3])  #Force
            pl.pause(1)

            outputNames = dataset.yHeaders
            mseDictionary = {}
            abeDictionary = {}
            for i in range(len(outputNames)):
                mse = mean_squared_error(actuals[:, i], predictions[:, i])  # Here <<<<<<
                abe = mean_absolute_error(actuals[:, i], predictions[:, i])
                print(f"{outputNames[i]}: MSE = {mse}, RMSE = {np.sqrt(mse)}")
                mseDictionary["mse_" + outputNames[i]] = mse
                print(f"{outputNames[i]}: ABE = {abe}")
                abeDictionary["abe_" + outputNames[i]] = abe

         #   wandb.log(mseDictionary)
         #   wandb.log(abeDictionary)
            # mse_x = mean_squared_error(actuals[:, 0], predictions[:, 0])
            # mse_y = mean_squared_error(actuals[:, 1], predictions[:, 1])
            # mse_z = mean_squared_error(actuals[:, 2], predictions[:, 2])
            # print(f"X: MSE = {mse_x}, RMSE = {np.sqrt(mse_x)}")
            # print(f"Y: MSE = {mse_y}, RMSE = {np.sqrt(mse_y)}")
            # print(f"Z: MSE = {mse_z}, RMSE = {np.sqrt(mse_z)}")
            # wandb.log({"mse_x": mse_x, "mse_y": mse_y, "mse_z": mse_z,
            #            "rmse_x": np.sqrt(mse_x), "rmse_y": np.sqrt(mse_y), "rmse_z": np.sqrt(mse_z)})

        # Save the model inside the files folder of the corresponding run in the wandb subdirectory. This means the
        # model .pth file will also be uploaded to wandb in the cloud so we won't accidentally delete it.
        datetime_stamp = datetime.fromtimestamp(wandb.run.start_time).strftime("%Y%m%d_%H%M%S")
        run_id = wandb.run.id
        save_path = f"model-{datetime_stamp}-{run_id}.pth"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, save_path))

        # End timing
        end_time = timeit.default_timer()
        print(f"Time taken: {end_time - start_time} seconds")

        # NOTE: Be sure to set plot to False in hyperparameter sweep configs, else your sweeps will be interrupted!
        # if config.plot:
            # plot_predictions(predictions, actuals, outputNames, unnormScaling=modelDataSummary.outputDataStatistics.getUnNormalizationScaling())
