import matplotlib.pyplot as plt


def plot_predictions(predictions, actuals, outputNames, unnormScaling=1):
    fig, axs = plt.subplots(len(outputNames), 1)
    for i in range(len(outputNames)):
        prediction_i = predictions[:, i]
        actual_i = actuals[:, i]
        axs[i].plot(actual_i)
        axs[i].plot(prediction_i)
        axs[i].set_ylabel(outputNames[i])

    plt.show()