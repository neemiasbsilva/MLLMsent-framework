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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-pb0m{border-color:inherit;text-align:center;vertical-align:bottom}
.tg .tg-za14{border-color:inherit;text-align:left;vertical-align:bottom}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-g7sd{border-color:inherit;font-weight:bold;text-align:left;vertical-align:middle}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-fll5{border-color:inherit;font-weight:bold;text-align:center;vertical-align:bottom}
.tg .tg-fgwq{background-color:#FFF;border-color:inherit;font-weight:bold;text-align:center;vertical-align:bottom}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-za14"></th>
    <th class="tg-za14"></th>
    <th class="tg-za14"></th>
    <th class="tg-uzvj" colspan="4"><span style="font-weight:bold">F1 - Score</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-g7sd"><span style="font-weight:bold">Type of dominance</span></td>
    <td class="tg-g7sd"><span style="font-weight:bold">Threshold (alpha)</span></td>
    <td class="tg-uzvj"><span style="font-weight:bold">Classification Problem</span></td>
    <td class="tg-uzvj"><span style="font-weight:bold">Percept Sent Paper</span></td>
    <td class="tg-uzvj"><span style="font-weight:bold">Zero-shot Vader</span></td>
    <td class="tg-uzvj"><span style="font-weight:bold">Mini-GPT4 + Zero-shot BART-LARGE-MNLI</span></td>
    <td class="tg-uzvj"><span style="font-weight:bold">Mini-GPT4 + Fine-Tuning BART-LARGE-MNLI</span></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Simple</td>
    <td class="tg-0pky" rowspan="4">alpha = 3</td>
    <td class="tg-pb0m">P5 (C=5)</td>
    <td class="tg-pb0m">45.00%</td>
    <td class="tg-pb0m">-</td>
    <td class="tg-pb0m">-</td>
    <td class="tg-fll5"><span style="font-weight:bold">49.54%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P3 (C=3)</td>
    <td class="tg-pb0m">61.00%</td>
    <td class="tg-pb0m">5.00%</td>
    <td class="tg-pb0m">54.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">70.44%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P2+ (C=2)</td>
    <td class="tg-pb0m">64.40%</td>
    <td class="tg-pb0m">47.00%</td>
    <td class="tg-pb0m">69.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">78.65%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P2- (C=2)</td>
    <td class="tg-pb0m">74.80%</td>
    <td class="tg-pb0m">40.00%</td>
    <td class="tg-pb0m">66.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">78.92%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Qualified</td>
    <td class="tg-0pky" rowspan="4">alpha = 4</td>
    <td class="tg-pb0m">P5 (C=5)</td>
    <td class="tg-pb0m">48.00%</td>
    <td class="tg-pb0m">-</td>
    <td class="tg-pb0m">-</td>
    <td class="tg-fll5"><span style="font-weight:bold">63.04%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P3 (C=3)</td>
    <td class="tg-pb0m">68.00%</td>
    <td class="tg-pb0m">2.00%</td>
    <td class="tg-pb0m">64.00%</td>
    <td class="tg-fgwq"><span style="font-weight:bold;background-color:#FFF">79.85%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P2+ (C=2)</td>
    <td class="tg-pb0m">76.60%</td>
    <td class="tg-pb0m">45.00%</td>
    <td class="tg-pb0m">74.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">85.09%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P2- (C=2)</td>
    <td class="tg-pb0m">84.00%</td>
    <td class="tg-pb0m">34.00%</td>
    <td class="tg-pb0m">72.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">86.58%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Absolute</td>
    <td class="tg-0pky" rowspan="4">alpha = 5</td>
    <td class="tg-pb0m">P5 (C=5)</td>
    <td class="tg-pb0m">51.00%</td>
    <td class="tg-pb0m">-</td>
    <td class="tg-pb0m">-</td>
    <td class="tg-fll5"><span style="font-weight:bold">76.05%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P3 (C=3)</td>
    <td class="tg-pb0m">69.00%</td>
    <td class="tg-pb0m">0.00%</td>
    <td class="tg-pb0m">75.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">90.91%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P2+ (C=2)</td>
    <td class="tg-pb0m">81.20%</td>
    <td class="tg-pb0m">45.00%</td>
    <td class="tg-pb0m">79.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">92.08%</span></td>
  </tr>
  <tr>
    <td class="tg-pb0m">P2- (C=2)</td>
    <td class="tg-pb0m">88.60%</td>
    <td class="tg-pb0m">27.00%</td>
    <td class="tg-pb0m">78.00%</td>
    <td class="tg-fll5"><span style="font-weight:bold">93.64%</span></td>
  </tr>
</tbody></table>
