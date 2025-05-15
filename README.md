# Emotion Classification Experiment

An emotion classification system based on the DeepSeek large language model, using the GoEmotions dataset to evaluate the impact of different prompt templates and batch sizes on emotion classification performance.

## Project Structure

```
DataMining
├── datasets/                        # Dataset directory
│   ├──goemotions/                   # Original GoEmotions dataset
│   └──prepared/                     # Official prepared dataset
├── deepseek_v3_tokenizer/           # DeepSeek tokenizer
├── visulaization/                   # Visualization results
├── data_prepare.py                  # Data preparation script
├── deepseek_emotion_classifier.py   # Main classifier implementation
├── emotions.py                      # Emotion label definitions
├── evaluation_utils.py              # Evaluation utility functions
├── prompt_utils.py                  # Prompt template utilities
├── README.md                        # Project documentation
├── requirements.txt                 # Project dependencies
└── tokenizer.py                     # Tokenizer utilities
```

## Environment Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Download GoEmotions Dataset (not used in pilot study)

Windows:
```bash
mkdir datasets/goemotions
curl -o ./datasets/goemotions/goemotions_1.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
curl -o ./datasets/goemotions/goemotions_2.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
curl -o ./datasets/goemotions/goemotions_3.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

Linux/Mac:
```bash
mkdir -p datasets/goemotions
curl -o ./datasets/goemotions/goemotions_1.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
curl -o ./datasets/goemotions/goemotions_2.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
curl -o ./datasets/goemotions/goemotions_3.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

### Prepare Experimental Data

Run the data preparation script to generate experimental data:

```bash
python data_prepare.py
```

This script will:
1. Load the offical prepared GoEmotions data
2. Filter multi-label data
3. Randomly select samples to generate an experimental dataset
4. Save example data for prompt examples

## Running Experiments

Run the emotion classification experiment, supporting different prompt templates and batch sizes:

```bash
python deepseek_emotion_classifier.py --prompt <template_name> --batch-size <batch_size>
```

Parameter description:
- `--prompt`: Prompt template, options: `minimal` (minimal template), `basic` (basic template), `rich_background` (rich background template)
- `--batch-size`: Batch size, such as 20, 50, 100

Examples:
```bash
# Use the basic template with batch size 20
python deepseek_emotion_classifier.py --prompt basic --batch-size 20

# Use the rich background template with batch size 50
python deepseek_emotion_classifier.py --prompt rich_background --batch-size 50

# Only display the prompt template without making API calls
python deepseek_emotion_classifier.py --prompt minimal --show-prompt
```

## Token Calculation

Calculate token consumption for different prompt templates and batch sizes:

```bash
python tokenizer.py
```

This script will generate a token consumption report saved in the `outputs` directory.

## Evaluation Metrics

The evaluation tools use multiple metrics to assess classification performance:
- Accuracy
- Precision
- Recall
- F1 score
- IoU score (Intersection over Union)

Evaluation results will be saved in the corresponding experiment folder in the `result` directory.

## Prompt Template Description

This project implements three prompt templates:
1. `minimal`: Minimal prompt template, containing only basic instructions
2. `basic`: Basic prompt template, standard version
3. `rich_background`: Rich background prompt template, containing extensive contextual information and detailed guidance

Detailed implementations of different templates can be viewed in `prompt_utils.py`.
