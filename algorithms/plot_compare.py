from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def load_data(file, what="test_rewards"):
    data = np.load(file, allow_pickle=True).item()
    # print(data.keys())
    return data[what]


test_interval = 100

savgol_frac = 0.15


def plot_compare(
    files,
    labels,
    xlabel,
    ylabel,
    title,
    save_path,
    smooth=True,
    test_data=True,
    yscale="linear",
    hline=None,
):
    savgol_length = int(len(load_data(files[0])) * savgol_frac)

    plt.figure()

    for file, label in zip(files, labels):
        data = np.array(load_data(file))
        if smooth:
            data = savgol_filter(data, savgol_length, 2, axis=0)
        if test_data:
            episodes = np.arange(0, len(data) * test_interval, test_interval)
            plt.plot(episodes, data, label=label)
        else:
            plt.plot(data, label=label)

    if hline is not None:
        plt.hlines(
            hline,
            0,
            len(data) * test_interval,
            colors="black",
            linestyles="dashed",
            zorder=-1,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.title(title)
    plt.legend(labels)
    plt.savefig(save_path)


if __name__ == "__main__":
    plot_compare(
        [
            "logs/lunar_tau_0_0004.npy",
            "logs/lunar_tau_0_001.npy",
            "logs/lunar_tau_0_005.npy",
            "logs/lunar_hard_50.npy",
            "logs/lunar_hard_100.npy",
        ],
        [
            r"$\tau = 0.0004$",
            r"$\tau = 0.001$",
            r"$\tau = 0.05$",
            "hard 50",
            "hard 100",
        ],
        "Episode",
        "Reward",
        "LunarLander-v2",
        "plots/lunar_tau.png",
        hline=200,
    )

    plot_compare(
        [
            "logs/pendulum_tau_0_0004.npy",
            "logs/pendulum_tau_0_001.npy",
            "logs/pendulum_tau_0_005.npy",
        ],
        [r"$\tau = 0.0004$", r"$\tau = 0.001$", r"$\tau = 0.05$"],
        "Episode",
        "Reward",
        "Pendulum-v0",
        "plots/pendulum_tau.png",
    )

    plot_compare(
        [
            "logs/lunar_an_0.0.npy",
            "logs/lunar_an_0.01.npy",
            "logs/lunar_an_0.002.npy",
            "logs/lunar_an_0.2.npy",
        ],
        [r"$\alpha = 0$", r"$\alpha = 0.01$", r"$\alpha = 0.002$", r"$\alpha = 0.2$"],
        "Episode",
        "Reward",
        "LunarLander-v2",
        "plots/lunar_an.png",
        hline=200,
    )

    plot_compare(
        [
            "logs/lunar_lander_td3.npy",
            "logs/lunar_lander_td3_no_double_q.npy",
        ],
        ["TD3", "TD3 no double Q"],
        "Episode",
        "Reward",
        "LunarLander-v2",
        "plots/lunar_td3.png",
        hline=200,
    )
