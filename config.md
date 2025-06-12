# Config

- Hardware: 8 * NVIDIA A100 40G
- batch_size_per_gpu: 2
- learning_rate: 3e-5

## Data

### llama2-7b

| Data Num. | ShareGPT | UltraChat | OpenThoughts2 |
| --------- | -------- | --------- | ------------- |
| 100k      | 8,670    | 57,785    | 33,545        |
| 200k      | 17,152   | 115,828   | 67,020        |
| 400k      | 33,992   | 231,476   | 134,532       |
| 600k      | 51,040   | 347,200   | 201,760       |
| 800k      | 68,000   | 463,000   | 269,000       |

## Model

### Eagle2

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

### HASS-1

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

### HASS-2

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

### Large Drafter, LD

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
