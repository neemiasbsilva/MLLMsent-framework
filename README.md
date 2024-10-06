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
    <th colspan="6">F1 - Score</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Type of dominance</td>
    <td>Threshold ($\sigma$)</td>
    <td>Classification Problem</td>
    <td>Percept Sent Paper</td>
    <td>MiniGPT-4 (Open Source) + <br>Vader<br></td>
    <td>MiniGPT-4 (Open Source) + <br>Zero-shot <br>BART-LARGE-MNLI</td>
    <td>MiniGPT-4 (Open Source) + <br>Fine-Tuning<br>DISTIL-BERT</td>
    <td>MiniGPT-4 (Open Source) +<br>Fine-Tuning<br>BART-LARGE-MNLI</td>
    <td>MiniGPT-4 (Open Source) + <br>Fine-Tuning<br>NV-Embed</td>
  </tr>
  <tr>
    <td rowspan="4">Simple</td>
    <td rowspan="4">$\sigma$ = 3</td>
    <td>P5 (C=5)</td>
    <td>45.00% [± 0.034]</td>
    <td>-</td>
    <td>23.04% [± 0.019]</td>
    <td>49.47% [± 0.018]</td>
    <td>47.97%</td>
    <td>49.54%</td>
  </tr>
  <tr>
    <td>P3 (C=3)</td>
    <td>61.00% [± 0.053]</td>
    <td>22.81% [± 0.014]</td>
    <td>29.71% [± 0.016]</td>
    <td>73.16% [± 0.025]</td>
    <td>68.77%</td>
    <td>70.44%</td>
  </tr>
  <tr>
    <td>P2+ (C=2)</td>
    <td>64.40% [± 0.062]</td>
    <td>48.75% [± 0.018]</td>
    <td>57.34% [± 0.023]</td>
    <td>79.18% [± 0.015]</td>
    <td>78.28%</td>
    <td>78.65%</td>
  </tr>
  <tr>
    <td>P2- (C=2)</td>
    <td>74.80% [± 0.050]</td>
    <td>28.58% [± 0.015]</td>
    <td>22.27% [± 0.014]</td>
    <td>81.68% [± 0.011]</td>
    <td>78.59%</td>
    <td>78.92%</td>
  </tr>
  <tr>
    <td rowspan="4">Qualified</td>
    <td rowspan="4">$\sigma$ = 4</td>
    <td>P5 (C=5)</td>
    <td>48.20% [± 0.089]</td>
    <td>-</td>
    <td>31.61% [± 0.036]</td>
    <td>68.64% [± 0.048]</td>
    <td>64.22%</td>
    <td>63.04%</td>
  </tr>
  <tr>
    <td>P3 (C=3)</td>
    <td>68.00% [± 0.052]</td>
    <td>2.34% [± 0.003]</td>
    <td>38.39% [± 0.025]</td>
    <td>82.25% [± 0.015]</td>
    <td>82.42%</td>
    <td>79.85%</td>
  </tr>
  <tr>
    <td>P2+ (C=2)</td>
    <td>76.60% [± 0.062]</td>
    <td>24.79% [± 0.016]</td>
    <td>36.85% [± 0.029]</td>
    <td>86.81% [± 0.013]</td>
    <td>85.98%</td>
    <td>85.09%</td>
  </tr>
  <tr>
    <td>P2- (C=2)</td>
    <td>84.00% [± 0.035]</td>
    <td>34.64% [± 0.008]</td>
    <td>69.48% [± 0.018]</td>
    <td>89.73% [± 0.016]</td>
    <td>86.97%</td>
    <td>86.58%</td>
  </tr>
  <tr>
    <td rowspan="4">Absolute</td>
    <td rowspan="4">$\sigma$ = 5</td>
    <td>P5 (C=5)</td>
    <td>51.20% [± 0.063]</td>
    <td>-</td>
    <td>47.10% [± 0.086]</td>
    <td>80.79% [± 0.069]</td>
    <td>72.69%</td>
    <td>76.05%</td>
  </tr>
  <tr>
    <td>P3 (C=3)</td>
    <td>69.00% [± 0.047]</td>
    <td>25.24% [± 0.037]</td>
    <td>36.93% [± 0.054]</td>
    <td>91.95% [± 0.019]</td>
    <td>91.15%</td>
    <td>90.91%</td>
  </tr>
  <tr>
    <td>P2+ (C=2)</td>
    <td>81.20% [± 0.054]</td>
    <td>25.24% [± 0.037]</td>
    <td>36.06% [± 0.046]</td>
    <td>92.88% [± 0.004]</td>
    <td>92.57%</td>
    <td>92.08%</td>
  </tr>
  <tr>
    <td>P2- (C=2)</td>
    <td>88.60% [± 0.011]</td>
    <td>29.76% [± 0.033]</td>
    <td>73.51% [± 0.040]</td>
    <td>93.75% [± 0.008]</td>
    <td>93.99%</td>
    <td>93.64%</td>
  </tr>
</tbody></table>