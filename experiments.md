# llama2-7b

Data distribution:

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

## Eagle2

**acceptance length $\tau$**

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

## HASS



## LD

