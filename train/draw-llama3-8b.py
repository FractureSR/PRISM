import numpy as np
from matplotlib import pyplot as plt

config = {
    "HASS-3": {
        "color": "#ff7f0e",
        "marker": "s"
    },
    "EAGLE3": {
        "color": "#2ca02c",
        "marker": "^"
    },
    "PRISM": {
        "color": "#d62728",
        "marker": "D"
    }
}

xticks = ["100k", "200k", "400k", "600k", "800k"]

exps = {
    "MT-bench": {
        "HASS-3": np.array([
            [4.56893, 4.96362, 5.15366, 5.20137, 5.24480],
            [4.13570, 4.47675, 4.68349, 4.76328, 4.80118]
        ]),
        "EAGLE3": np.array([
            [4.83510, 5.09287, 5.32010, 5.36137, 5.41319],
            [4.42497, 4.66181, 4.86650, 4.92450, 4.89523]
        ]),
        "PRISM": np.array([
            [5.14252, 5.27539, 5.43683, 5.51294, 5.55223],
            [4.66488, 4.76437, 4.96659, 5.06792, 4.98788]
        ])
    },
    "HumanEval": {
        "HASS-3": np.array([
            [5.57321, 5.83132, 6.06818, 6.09143, 6.12733],
            [5.32623, 5.66073, 5.81162, 5.82378, 5.85317]
        ]),
        "EAGLE3": np.array([
            [5.84186, 5.99639, 6.16379, 6.21239, 6.25450],
            [5.53334, 5.63587, 5.83999, 5.99121, 6.00221]
        ]),
        "PRISM": np.array([
            [6.16391, 6.26005, 6.38158, 6.43769, 6.46091],
            [5.82073, 6.00615, 6.07868, 6.11884, 6.20578]
        ])
    },
    "GSM8K": {
        "HASS-3": np.array([
            [5.14749, 5.64556, 5.83745, 5.91626, 5.91216],
            [4.87022, 5.42925, 5.66350, 5.68835, 5.74633]
        ]),
        "EAGLE3": np.array([
            [5.38041, 5.74095, 5.90802, 6.06136, 6.08128],
            [5.19030, 5.53035, 5.65158, 5.81728, 5.84843]
        ]),
        "PRISM": np.array([
            [5.86677, 6.00308, 6.14677, 6.21298, 6.23580],
            [5.57597, 5.71765, 5.96026, 6.03565, 5.97620]
        ])
    },
    "Alpaca": {
        "HASS-3": np.array([
            [4.78656, 5.17630, 5.43573, 5.44340, 5.43318],
            [4.40090, 4.84232, 5.05960, 5.06703, 5.09216]
        ]),
        "EAGLE3": np.array([
            [5.13071, 5.39130, 5.53601, 5.59397, 5.64190],
            [4.86603, 4.90671, 5.07212, 5.17077, 5.33028]
        ]),
        "PRISM": np.array([
            [5.29137, 5.43755, 5.60986, 5.65036, 5.71664],
            [4.99205, 5.09049, 5.27211, 5.21822, 5.28823]
        ])
    },
    "CNN/DM": {
        "HASS-3": np.array([
            [4.15562, 4.66771, 4.94444, 4.88727, 5.13111],
            [3.86624, 4.25736, 4.54411, 4.43344, 4.61062]
        ]),
        "EAGLE3": np.array([
            [4.28696, 4.41948, 4.71347, 4.85753, 4.83344],
            [3.88663, 4.12648, 4.33049, 4.41356, 4.44261]
        ]),
        "PRISM": np.array([
            [4.87432, 5.07487, 5.27624, 5.38739, 5.42418],
            [4.41422, 4.69108, 4.79490, 4.85537, 4.89076]
        ])
    },
    "Natural Ques.": {
        "HASS-3": np.array([
            [3.63343, 4.09168, 4.32561, 4.28565, 4.33502],
            [3.46048, 3.75132, 4.02459, 4.15171, 4.04818]
        ]),
        "EAGLE3": np.array([
            [3.92492, 4.15292, 4.31472, 4.43129, 4.49520],
            [3.63554, 3.82915, 3.97047, 4.15073, 4.18076]
        ]),
        "PRISM": np.array([
            [4.24841, 4.46707, 4.61694, 4.65323, 4.71661],
            [3.91144, 4.17247, 4.21648, 4.31143, 4.33946]
        ])
    }
}

means = {
    "HASS-3": np.zeros((2, 5)),
    "EAGLE3": np.zeros((2, 5)),
    "PRISM": np.zeros((2, 5))
}

for results in exps.values():
    for model, values in results.items():
        means[model] += values / 6.0

for temperature in [0, 1]:
    plt.figure(figsize=(11, 7))

    for model, values in means.items():
        plt.plot(
            range(1, 6),
            values[temperature],
            color=config[model]["color"],
            marker=config[model]["marker"],
            linestyle="-" if temperature == 0 else ":"
        )

    plt.grid()
    plt.legend(["HASS-3", "EAGLE3", "PRISM"], loc="lower right", fontsize=12)
    plt.title(f"Temperature = {temperature}", fontsize=18)

    plt.xticks(range(1, 6), xticks, fontsize=12)
    plt.xlabel("Train Data Volume", fontsize=15)

    if temperature == 0:
        plt.yticks(np.arange(4.6, 5.7, 0.1), fontsize=12)
    else:
        plt.yticks(np.arange(4.3, 5.4, 0.1), fontsize=12)
    plt.ylabel("Acceptance Length", fontsize=15)
