# Large Drafter, LD

## llama2-7b

| Data Num. | ShareGPT | UltraChat | OpenThoughts2 |
| --------- | -------- | --------- | ------------- |
| 100k      | 8,670    | 57,785    | 33,545        |
| 200k      | 17,152   | 115,828   | 67,020        |
| 400k      | 33,992   | 231,476   | 134,532       |
| 600k      | 51,040   | 347,200   | 201,760       |
| 800k      | 68,000   | 463,000   | 269,000       |

- Hardware: 8 * NVIDIA A100 40G
- batch_size_per_gpu: 2
- learning_rate: 3e-5

### Eagle2

#### config

```json
{
  "eagle_config": {
    "num_hidden_layers": 1
  },
  "num_steps": 1,
  "num_step_models": 1,
  "step_mapping": {
    "0": 0
  }
}
```

#### acceptance length

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

#### speedup ratio

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

### HASS-1

#### config

```json
{
  "eagle_config": {
    "num_hidden_layers": 1
  },
  "num_steps": 3,
  "num_step_models": 1,
  "step_mapping": {
    "0": 0,
    "1": 0,
    "2": 0
  }
}
```

#### acceptance length

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

#### speedup ratio

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

### HASS-2

#### config

```json
{
  "eagle_config": {
    "num_hidden_layers": 2
  },
  "num_steps": 3,
  "num_step_models": 1,
  "step_mapping": {
    "0": 0,
    "1": 0,
    "2": 0
  }
}
```

#### acceptance length

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

#### speedup ratio

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

### LD

#### config

```json
{
  "eagle_config": {
    "num_hidden_layers": 1
  },
  "num_steps": 3,
  "num_step_models": 2,
  "step_mapping": {
    "0": 0,
    "1": 1,
    "2": 1
  }
}
```

#### acceptance length

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

#### speedup ratio

`temperature = 0`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

`temperature = 1`

| Data Num.       | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Ques. | Mean |
| --------------- | -------- | --------- | ----- | ------ | ------ | ------------- | ---- |
| 100k (epoch=40) |          |           |       |        |        |               |      |
| 200k (epoch=30) |          |           |       |        |        |               |      |
| 400k (epoch=25) |          |           |       |        |        |               |      |
| 600k (epoch=20) |          |           |       |        |        |               |      |
| 800k (epoch=15) |          |           |       |        |        |               |      |

