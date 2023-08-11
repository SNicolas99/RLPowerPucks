from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from plot_compare import load_data


def plot_hockey(
    file,
    xlabel,
    ylabel,
    title,
    save_path,
    smooth=True,
    test_data=True,
    yscale="linear",
    whats=["winrate", "lossrate"],
):
    savgol_frac = 0.3

    plt.figure()

    for what in whats:
        data = np.array(load_data(file, what=what))
        savgol_length = int(len(data) * savgol_frac)
        if smooth:
            data = savgol_filter(data, savgol_length, 2, axis=0)

        if test_data:
            episodes = np.arange(0, len(data) * 100, 100)
        else:
            episodes = np.arange(0, len(data))

        plt.plot(episodes, data, label=what)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)


name = "hockey_strong_mixed"

plot_hockey(
    f"logs/{name}.npy",
    xlabel="Episodes",
    ylabel="Winrate",
    title="Hockey",
    save_path=f"plots/{name}.png",
    smooth=True,
    test_data=True,
    yscale="linear",
    whats=["winrate", "lossrate"],
)
