import numpy as np
from matplotlib import pyplot as plt

config = {
    'Eagle2': {
        'color': '#1f77b4',
        'marker': 'o'
    },
    'HASS-1': {
        'color': '#ff7f0e',
        'marker': 's'
    },
    'HASS-2': {
        'color': '#2ca02c',
        'marker': '^'
    },
    'LD': {
        'color': '#d62728',
        'marker': 'D'
    }
}

xticks = ['100k', '200k', '400k', '600k', '800k']

exps = {
    'MT-bench': {
        'Eagle2': [
            [4.73315, 4.89139, 4.98708, 5.04436, 5.02846],
            [4.51329, 4.70636, 4.77445, 4.84118, 4.82961]
        ],
        'HASS-1': [
            [5.09173, 5.23063, 5.3507, 5.33786, 5.34127],
            [4.86165, 4.94343, 5.04385, 5.06115, 5.09524]
        ],
        'HASS-2': [
            [4.91999, np.nan, np.nan, np.nan, np.nan],
            [4.65474, np.nan, np.nan, np.nan, np.nan]
        ],
        'LD': [
            [4.72376, np.nan, np.nan, np.nan, np.nan],
            [4.5262, np.nan, np.nan, np.nan, np.nan]
        ]
    },
    'HumanEval': {
        'Eagle2': [
            [5.35449, 5.53025, 5.61162, 5.64755, 5.62172],
            [5.07249, 5.25476, 5.24275, 5.29181, 5.30573]
        ],
        'HASS-1': [
            [5.78596, 5.86957, 5.9641, 5.99358, 5.99337],
            [5.37265, 5.587, 5.66884, 5.64579, 5.58696]
        ],
        'HASS-2': [
            [5.58743, np.nan, np.nan, np.nan, np.nan],
            [5.22287, np.nan, np.nan, np.nan, np.nan]
        ],
        'LD': [
            [5.41697, np.nan, np.nan, np.nan, np.nan],
            [5.17176, np.nan, np.nan, np.nan, np.nan]
        ]
    },
    'GSM8K': {
        'Eagle2': [
            [4.84016, 4.99998, 5.04102, 5.0628, 5.1267],
            [4.87757, 4.87929, 5.05543, 5.11579, 5.16248]
        ],
        'HASS-1': [
            [5.22653, 5.30258, 5.39044, 5.42064, 5.41632],
            [5.17258, 5.32093, 5.24608, 5.38297, 5.31655]
        ],
        'HASS-2': [
            [4.97048, np.nan, np.nan, np.nan, np.nan],
            [4.89504, np.nan, np.nan, np.nan, np.nan]
        ],
        'LD': [
            [4.79334, np.nan, np.nan, np.nan, np.nan],
            [4.7594, np.nan, np.nan, np.nan, np.nan]
        ]
    },
    'Alpaca': {
        'Eagle2': [
            [4.62961, 4.80094, 4.94934, 4.93946, 4.96793],
            [4.54291, 4.54484, 4.75235, 4.76099, 4.81447]
        ],
        'HASS-1': [
            [4.96695, 5.11803, 5.20702, 5.24013, 5.26022],
            [4.77251, 4.94477, 5.03474, 5.18521, 5.14378]
        ],
        'HASS-2': [
            [4.71984, np.nan, np.nan, np.nan, np.nan],
            [4.6133, np.nan, np.nan, np.nan, np.nan]
        ],
        'LD': [
            [4.59303, np.nan, np.nan, np.nan, np.nan],
            [4.4966, np.nan, np.nan, np.nan, np.nan]
        ]
    },
    'CNN/DM': {
        'Eagle2': [
            [4.31285, 4.4856, 4.57852, 4.61813, 4.63727],
            [4.13297, 4.29954, 4.36809, 4.45527, 4.40932]
        ],
        'HASS-1': [
            [4.67958, 4.85362, 4.94466, 4.95558, 4.98692],
            [4.44409, 4.61786, 4.67893, 4.76209, 4.72477]
        ],
        'HASS-2': [
            [4.49193, np.nan, np.nan, np.nan, np.nan],
            [4.28667, np.nan, np.nan, np.nan, np.nan]
        ],
        'LD': [
            [4.37033, np.nan, np.nan, np.nan, np.nan],
            [4.08461, np.nan, np.nan, np.nan, np.nan]
        ]
    },
    'Natural Ques.': {
        'Eagle2': [
            [4.25697, 4.4412, 4.54477, 4.56902, 4.57432],
            [4.11488, 4.23569, 4.39021, 4.40386, 4.43694]
        ],
        'HASS-1': [
            [4.54263, 4.74907, 4.84555, 4.87466, 4.91995],
            [4.45015, 4.55391, 4.69417, 4.74354, 4.7457]
        ],
        'HASS-2': [
            [4.27417, np.nan, np.nan, np.nan, np.nan],
            [4.15626, np.nan, np.nan, np.nan, np.nan]
        ],
        'LD': [
            [4.12356, np.nan, np.nan, np.nan, np.nan],
            [4.01658, np.nan, np.nan, np.nan, np.nan]
        ]
    }
}

for benchmark in exps.keys():
    plt.figure(figsize=(8, 6))

    values = []
    legends = []
    for model in exps[benchmark].keys():
        plt.plot(
            range(1, 6),
            exps[benchmark][model][0],
            color=config[model]['color'],
            marker=config[model]['marker'],
            linestyle='-'
        )
        values.extend(exps[benchmark][model][0])
        legends.append(f'{model} (t=0)')

        plt.plot(
            range(1, 6),
            exps[benchmark][model][1],
            color=config[model]['color'],
            marker=config[model]['marker'],
            linestyle=':'
        )
        values.extend(exps[benchmark][model][1])
        legends.append(f'{model} (t=1)')

    plt.grid()
    plt.legend(legends, loc='lower right')
    plt.title(benchmark, fontsize=20)
    plt.xticks(range(1, 6), xticks, fontsize=12)
    plt.yticks(np.arange(round(min(values), 1), round(max(values), 1), 0.1), fontsize=12)
    plt.ylabel('Acceptance Length', fontsize=16)
