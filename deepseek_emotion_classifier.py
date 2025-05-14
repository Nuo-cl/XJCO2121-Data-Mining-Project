import os
import json
import numpy as np
from openai import OpenAI
from datasets import Dataset
from tqdm import tqdm
import evaluate
import time
from datetime import datetime
from pathlib import Path
import shutil
from sklearn.metrics import precision_recall_fscore_support
import argparse
# Import evaluation functions from evaluation_utils.py
from evaluation_utils import evaluate_predictions
# Import prompt generation functions
from prompt_utils import prompt_templates

# Load dataset
def load_dataset(data_path, labels_path):
    """Load data and labels from files, and build a dataset using the datasets library"""
    with open(data_path, 'r', encoding='utf-8') as f:
        texts = f.read().splitlines()
    
    labels = np.load(labels_path)
    
    return Dataset.from_dict({
        'text': texts,
        'labels': labels.tolist()
    })

# Get emotion_dict
from emotions import emotion_dict

# Reverse mapping, from number ID to emotion label
id_to_emotion = {v: k for k, v in emotion_dict.items()}

# Load example data
def load_example_data():
    """Load example data as part of the prompt"""
    examples = []
    
    # Read from new example.txt file format where each line is <data>:<label>
    example_file = 'datasets/pilot_study/example.txt'
    with open(example_file, 'r', encoding='utf-8') as f:
        example_lines = f.read().splitlines()
    
    # Parse each line to extract text and label
    for line in example_lines:
        # Find the last colon in the line
        last_colon_idx = line.rfind(':')
        if last_colon_idx != -1:
            text = line[:last_colon_idx].strip()
            # Convert label from string to integer
            try:
                label = int(line[last_colon_idx+1:].strip())
                examples.append({
                    'text': text,
                    'label': label
                })
            except ValueError:
                print(f"Warning: Could not parse label in line: {line}")
    
    print(f"Loaded {len(examples)} example items from {example_file}")
    return examples

# Call DeepSeek API using OpenAI library
def call_deepseek_api(prompt):
    """Call DeepSeek API to get emotion classification results using OpenAI library"""
    # Configure the client with DeepSeek API base URL
    client = OpenAI(
        api_key='sk-9da535bc9b544bdda2418983dc6e3666',
        base_url="https://api.deepseek.com/v1"  # DeepSeek API base URL
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",  # Replace with the actual model being used
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retrying
            else:
                raise

# Parse API response, extract prediction results
def parse_response(response_text, batch_texts):
    """Extract prediction results from API response"""
    # Find the part with %%% markers
    start_marker = "%%%"
    end_marker = "%%%"
    
    start_idx = response_text.find(start_marker)
    end_idx = response_text.rfind(end_marker)
    
    if start_idx == -1 or end_idx == -1 or start_idx == end_idx:
        print(f"Cannot find markers in the response. Response content:\n{response_text}")
        return None
    
    result_text = response_text[start_idx + len(start_marker):end_idx].strip()
    
    # Try to parse the result as a JSON object
    try:
        import json
        predictions_dict = json.loads(result_text)
        
        # Extract predictions in the same order as batch_texts
        predictions = []
        missing_texts = []
        
        for text in batch_texts:
            if text in predictions_dict:
                predictions.append(int(predictions_dict[text]))
            else:
                # Try with truncated text as key (in case of very long texts)
                found = False
                for key in predictions_dict.keys():
                    # Check if text starts with key or key starts with text
                    if text.startswith(key) or key.startswith(text):
                        predictions.append(int(predictions_dict[key]))
                        found = True
                        break
                
                if not found:
                    missing_texts.append(text)
                    # Default to "neutral" category if text is missing
                    predictions.append(emotion_dict.get("neutral", 27))
        
        if missing_texts:
            print(f"Warning: {len(missing_texts)} texts were not found in the prediction results.")
            if len(missing_texts) < 5:  # Only print a few examples
                for text in missing_texts[:3]:
                    print(f"  Missing text: '{text[:50]}...'")
            else:
                print("  Too many missing texts to display.")
        
        return predictions
    except Exception as e:
        print(f"Failed to parse prediction results as JSON: {e}")
        print(f"Original result text: {result_text}")
        
        # Fallback: try to parse as the old format [num,num,...]
        try:
            # Handle potential format issues
            result_text = result_text.replace(" ", "")
            if result_text.startswith("[") and result_text.endswith("]"):
                result_text = result_text[1:-1]
            
            predictions = [int(x) for x in result_text.split(",")]
            print("Successfully parsed using fallback method.")
            return predictions
        except:
            print("Fallback parsing also failed.")
            return None

# Call API with verification for batch size
def call_api_with_verification(batch_texts, examples, prompt_template):
    """Call API and verify that the number of predictions matches the batch size"""
    max_retries = 5
    for attempt in range(max_retries):
        # Build prompt using the selected template
        prompt = prompt_template(batch_texts, examples, emotion_dict)
        
        # Call API
        print(f"API request attempt {attempt+1}/{max_retries}...")
        response = call_deepseek_api(prompt)
        
        # Parse results with the new JSON format
        predictions = parse_response(response, batch_texts)
        
        # Verify predictions count matches batch size
        if predictions is not None:
            if len(predictions) == len(batch_texts):
                print(f"Successfully received {len(predictions)} predictions matching batch size")
                return predictions
            else:
                print(f"Mismatch between predictions count ({len(predictions)}) and batch size ({len(batch_texts)}). Retrying...")
        
        # If we get here, either predictions is None or the counts don't match
        if attempt < max_retries - 1:
            print(f"Waiting before retry...")
            time.sleep(10)  # Longer wait between retries
    
    print(f"Failed to get correct predictions after {max_retries} attempts.")
    return None

# Create result directory with prompt type and batch size
def create_result_directory(prompt_type, batch_size):
    """Create a directory for storing results based on prompt type and batch size"""
    # Create base directory if it doesn't exist
    base_dir = Path("./result")
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectory with prompt type and batch size
    dir_name = f"{prompt_type}_batchsize{batch_size}"
    result_dir = base_dir / dir_name
    
    # Check if directory already exists, if so add timestamp to avoid overwriting
    if result_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{prompt_type}_batchsize{batch_size}_{timestamp}"
        result_dir = base_dir / dir_name
    
    result_dir.mkdir(exist_ok=True)
    
    return result_dir

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DeepSeek Emotion Classification')
    
    # Add argument for prompt template selection
    parser.add_argument(
        '--prompt', 
        type=str, 
        default='basic',
        choices=list(prompt_templates.keys()),
        help=f'Prompt template to use. Available options: {", ".join(prompt_templates.keys())}'
    )
    
    # Add argument for batch size
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of texts to process in each batch'
    )
    
    # Add argument to display prompt and exit
    parser.add_argument(
        '--show-prompt',
        action='store_true',
        help='Show the prompt template and exit without making API calls'
    )
    
    return parser.parse_args()

# Main function
def main():
    # Parse command line arguments
    args = parse_arguments()
    # Select prompt template
    prompt_template = prompt_templates[args.prompt]
    print(f"Using prompt template: {args.prompt}")
    
    # Create result directory with prompt type and batch size
    result_dir = create_result_directory(args.prompt, args.batch_size)
    print(f"Results will be saved to {result_dir}")
    
    # Save the prompt template name to the result directory
    with open(result_dir / "prompt_template.txt", 'w') as f:
        f.write(f"Prompt template: {args.prompt}\n")
        f.write(f"Batch size: {args.batch_size}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        'datasets/pilot_study/data.txt',
        'datasets/pilot_study/labels.npy'
    )
    print(f"Loaded {len(dataset)} data entries")
    
    # Load example data
    print("Loading example data...")
    examples = load_example_data()
    
    # If show-prompt flag is set, display an example prompt and exit
    if args.show_prompt:
        # Get a small subset of the dataset for display
        sample_texts = dataset['text'][:5]
        prompt = prompt_template(sample_texts, examples, emotion_dict)
        print("\nExample prompt with 5 sample texts:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        return
    
    # Process data in batches
    batch_size = args.batch_size
    all_predictions = []
    
    print(f"Starting batch processing, {batch_size} entries per batch")
    for i in tqdm(range(0, len(dataset), batch_size)):
        # Get current batch texts
        batch_texts = dataset['text'][i:i+batch_size]
        
        # Call API with verification and retry mechanism
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}...")
        batch_predictions = call_api_with_verification(batch_texts, examples, prompt_template)
        
        if batch_predictions:
            all_predictions.extend(batch_predictions)
            print(f"Successfully added batch results, accumulated {len(all_predictions)} predictions")
        else:
            print(f"Batch {i//batch_size + 1} failed after all retries. Continuing to next batch...")
    
    # Save intermediate predictions to result directory
    pred_path = result_dir / "deepseek_predictions.npy"
    np.save(pred_path, np.array(all_predictions))
    print(f"Prediction results saved to {pred_path}")
    
    # Copy original data files to result directory for reference
    shutil.copy('datasets/pilot_study/data.txt', result_dir / "data.txt")
    shutil.copy('datasets/pilot_study/labels.npy', result_dir / "labels.npy")
    
    # Evaluate results
    if len(all_predictions) == len(dataset):
        # Evaluate with multiple metrics
        print("Evaluating predictions...")
        results = evaluate_predictions(all_predictions, dataset['labels'], id_to_emotion)
        
        # Print summary of results
        print("\nEvaluation results summary:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Mean IoU: {results['iou_scores']['mean_iou']:.4f}")
        
        # Save full evaluation results to result directory
        eval_path = result_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed evaluation results saved to {eval_path}")
        
        # Create a more readable summary file
        summary_path = result_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("EMOTION CLASSIFICATION EVALUATION SUMMARY\n")
            f.write("=======================================\n\n")
            f.write(f"Prompt template: {args.prompt}\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1 Score: {results['f1']:.4f}\n")
            f.write(f"Mean IoU: {results['iou_scores']['mean_iou']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS (Top 5 by F1 Score):\n")
            f.write("------------------------------------\n")
            # Sort emotions by F1 score
            sorted_emotions = sorted(
                results['class_metrics'].items(),
                key=lambda x: x[1]['f1'],
                reverse=True
            )
            for emotion, metrics in sorted_emotions[:5]:
                f.write(f"{emotion}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"  IoU: {results['iou_scores'].get(emotion, 0):.4f}\n\n")
        
        print(f"Summary report saved to {summary_path}")
    else:
        print(f"Warning: Number of predictions ({len(all_predictions)}) does not match dataset size ({len(dataset)}), cannot evaluate")

if __name__ == "__main__":
    main() 