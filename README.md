# PerceptSent: an approach using LLMs to sentiment analysis

## Project Structure

```
project_root/
│
├── data/                # Data directory
│   ├── train/           # Training data
│   ├── validation/      # Validation data
│   └── test/            # Test data
│
├── models/              # Model definitions
│   ├── __init__.py
│   ├── model.py         # Define your neural network architectures
│   └── utils.py         # Utility functions related to models
│
├── utils/               # Utility functions and scripts
│   ├── __init__.py
│   ├── data_loader.py   # Custom data loader
│   ├── visualization.py # Functions for visualizing data, results, etc.
│   └── other_utils.py   # Other utility functions
│
├── experiments/         # Experiment configurations and logs
│   ├── experiment1/
│   │   ├── config.yaml  # Configuration for experiment 1
│   │   └── logs/        # Logs for experiment 1
│   ├── experiment2/
│   │   ├── config.yaml  # Configuration for experiment 2
│   │   └── logs/        # Logs for experiment 2
│   └── ...
│
├── checkpoints/         # Saved model checkpoints
│
├── scripts/             # Training, testing, and evaluation scripts
│   ├── train.py         # Script for training the models
│   ├── test.py          # Script for testing the models
│   └── evaluate.py      # Script for evaluation
│
├── requirements.txt     # Python dependencies
├── README.md            # Project description and instructions
└── .gitignore           # Specify files/folders to be ignored by version control

```


## Experiments & Results

<table><thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="4">F1 - Score</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Type of dominance</td>
    <td>Threshold (alpha)</td>
    <td>Classification Problem</td>
    <td>Percept Sent Paper</td>
    <td>Zero-shot Vader</td>
    <td>Mini-GPT4 + Zero-shot BART-LARGE-MNLI</td>
    <td>Mini-GPT4 + Fine-Tuning BART-LARGE-MNLI</td>
  </tr>
  <tr>
    <td rowspan="4">Simple</td>
    <td rowspan="4">alpha = 3</td>
    <td>P5 (C=5)</td>
    <td>45.00%</td>
    <td>-</td>
    <td>-</td>
    <td>49.54%</td>
  </tr>
  <tr>
    <td>P3 (C=3)</td>
    <td>61.00%</td>
    <td>5.00%</td>
    <td>54.00%</td>
    <td>70.44%</td>
  </tr>
  <tr>
    <td>P2+ (C=2)</td>
    <td>64.40%</td>
    <td>47.00%</td>
    <td>69.00%</td>
    <td>78.65%</td>
  </tr>
  <tr>
    <td>P2- (C=2)</td>
    <td>74.80%</td>
    <td>40.00%</td>
    <td>66.00%</td>
    <td>78.92%</td>
  </tr>
  <tr>
    <td rowspan="4">Qualified</td>
    <td rowspan="4">alpha = 4</td>
    <td>P5 (C=5)</td>
    <td>48.00%</td>
    <td>-</td>
    <td>-</td>
    <td>63.04%</td>
  </tr>
  <tr>
    <td>P3 (C=3)</td>
    <td>68.00%</td>
    <td>2.00%</td>
    <td>64.00%</td>
    <td>79.85%</td>
  </tr>
  <tr>
    <td>P2+ (C=2)</td>
    <td>76.60%</td>
    <td>45.00%</td>
    <td>74.00%</td>
    <td>85.09%</td>
  </tr>
  <tr>
    <td>P2- (C=2)</td>
    <td>84.00%</td>
    <td>34.00%</td>
    <td>72.00%</td>
    <td>86.58%</td>
  </tr>
  <tr>
    <td rowspan="4">Absolute</td>
    <td rowspan="4">alpha = 5</td>
    <td>P5 (C=5)</td>
    <td>51.00%</td>
    <td>-</td>
    <td>-</td>
    <td>76.05%</td>
  </tr>
  <tr>
    <td>P3 (C=3)</td>
    <td>69.00%</td>
    <td>0.00%</td>
    <td>75.00%</td>
    <td>90.91%</td>
  </tr>
  <tr>
    <td>P2+ (C=2)</td>
    <td>81.20%</td>
    <td>45.00%</td>
    <td>79.00%</td>
    <td>92.08%</td>
  </tr>
  <tr>
    <td>P2- (C=2)</td>
    <td>88.60%</td>
    <td>27.00%</td>
    <td>78.00%</td>
    <td>93.64%</td>
  </tr>
</tbody></table>