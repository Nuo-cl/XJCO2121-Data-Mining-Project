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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

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

# Build prompt
def build_prompt(batch_texts, examples):
    """Build prompt to send to DeepSeek API"""
    emotion_mapping = "Emotion mapping table:\n"
    for emotion, idx in emotion_dict.items():
        emotion_mapping += f"{emotion}: {idx}\n"
    
    examples_text = "Here are example data and their corresponding emotion labels:\n"
    for ex in examples:
        examples_text += f"Text: {ex['text']}\n"
        examples_text += f"Label: {ex['label']}\n"
        examples_text += "---\n"
    
    batch_data = "Please classify the following texts with their corresponding emotion labels:\n"
    for i, text in enumerate(batch_texts):
        batch_data += f"{i+1}. {text}\n"
    
    instruction = """You are a text emotion classification assistant. I will provide some texts, and you'll classify each text's emotion.
Please strictly follow the format [number,number,...] for classification results, without any other explanation.
In your response, the classification result section needs to be marked with %%% at the beginning and end."""
    
    prompt = f"{instruction}\n\n{emotion_mapping}\n\n{examples_text}\n\n{batch_data}\n\nPlease provide the results in the exact format [number,number,...] and use %%% as markers."
    
    return prompt

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
def parse_response(response_text):
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
    
    # Try to parse the result as a list of numbers
    try:
        # Handle potential format issues
        result_text = result_text.replace(" ", "")
        if result_text.startswith("[") and result_text.endswith("]"):
            result_text = result_text[1:-1]
        
        predictions = [int(x) for x in result_text.split(",")]
        return predictions
    except Exception as e:
        print(f"Failed to parse prediction results: {e}")
        print(f"Original result text: {result_text}")
        return None

# Call API with verification for batch size
def call_api_with_verification(batch_texts, examples):
    """Call API and verify that the number of predictions matches the batch size"""
    max_retries = 5
    for attempt in range(max_retries):
        # Build prompt
        prompt = build_prompt(batch_texts, examples)
        
        # Call API
        print(f"API request attempt {attempt+1}/{max_retries}...")
        response = call_deepseek_api(prompt)
        
        # Parse results
        predictions = parse_response(response)
        
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

# Calculate IoU (Intersection over Union) for each class
def calculate_iou(pred_labels, true_labels, num_classes=28):
    """Calculate IoU for each class"""
    # Create one-hot encoded vectors for predictions and true labels
    pred_one_hot = np.zeros((len(pred_labels), num_classes))
    true_one_hot = np.zeros((len(true_labels), num_classes))
    
    for i, pred in enumerate(pred_labels):
        pred_one_hot[i, pred] = 1
    
    for i, true in enumerate(true_labels):
        true_one_hot[i, true] = 1
    
    # Calculate IoU for each class
    iou_scores = {}
    for class_idx in range(num_classes):
        # Count true positives, false positives, and false negatives
        intersection = np.logical_and(pred_one_hot[:, class_idx], true_one_hot[:, class_idx]).sum()
        union = np.logical_or(pred_one_hot[:, class_idx], true_one_hot[:, class_idx]).sum()
        
        # Calculate IoU for this class (avoiding division by zero)
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        # Store the IoU score with the emotion name
        emotion_name = id_to_emotion.get(class_idx, f"Unknown-{class_idx}")
        iou_scores[emotion_name] = float(iou)
    
    # Calculate mean IoU across all classes
    iou_scores["mean_iou"] = float(np.mean(list(iou_scores.values())))
    
    return iou_scores

# Evaluate predictions with multiple metrics
def evaluate_predictions(predictions, references):
    """Evaluate predictions using multiple metrics"""
    results = {}
    
    # Get unique labels that actually appear in the data
    unique_labels = sorted(set(references).union(set(predictions)))
    
    # Accuracy
    accuracy_metric = evaluate.load("accuracy")
    accuracy_results = accuracy_metric.compute(predictions=predictions, references=references)
    results["accuracy"] = accuracy_results["accuracy"]
    
    # Precision, Recall, F1 Score - with zero_division=0 to handle warnings
    precision, recall, f1, support = precision_recall_fscore_support(
        references, predictions, average="macro", zero_division=0,
        labels=unique_labels  # Only use labels that appear in the data
    )
    results["precision"] = float(precision)
    results["recall"] = float(recall)
    results["f1"] = float(f1)
    
    # Per-class precision, recall, and F1
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        references, predictions, average=None, zero_division=0,
        labels=unique_labels  # Only use labels that appear in the data
    )
    
    # Add per-class metrics
    class_metrics = {}
    for i, label in enumerate(unique_labels):
        emotion_name = id_to_emotion.get(label, f"Unknown-{label}")
        class_metrics[emotion_name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i])
        }
    results["class_metrics"] = class_metrics
    
    # IoU scores - make sure to only calculate for labels that appear
    iou_scores = calculate_iou(predictions, references, num_classes=max(unique_labels) + 1)
    # Filter IoU scores to only include emotions that appear in the data
    filtered_iou_scores = {}
    for label in unique_labels:
        emotion_name = id_to_emotion.get(label, f"Unknown-{label}")
        if emotion_name in iou_scores:
            filtered_iou_scores[emotion_name] = iou_scores[emotion_name]
    # Add mean IoU
    filtered_iou_scores["mean_iou"] = float(np.mean(list(filtered_iou_scores.values())))
    results["iou_scores"] = filtered_iou_scores
    
    # Confusion Matrix (store as list of lists for JSON serialization)
    cm = confusion_matrix(references, predictions, labels=unique_labels).tolist()
    results["confusion_matrix"] = cm
    
    # Get emotion names for labels that appear in the data
    present_emotion_names = [id_to_emotion.get(label, f"Unknown-{label}") for label in unique_labels]
    
    # Classification report
    try:
        report = classification_report(
            references, 
            predictions,
            labels=unique_labels,  # Explicitly specify which labels to include
            target_names=present_emotion_names,  # Use names only for labels that appear
            output_dict=True,
            zero_division=0  # Handle division by zero gracefully
        )
        results["classification_report"] = report
    except Exception as e:
        print(f"Warning: Could not generate classification report: {e}")
        results["classification_report"] = {"error": str(e)}
    
    # Add metadata about the evaluation
    results["metadata"] = {
        "num_unique_labels": len(unique_labels),
        "labels_present": unique_labels,
        "total_samples": len(predictions)
    }
    
    return results

# Create result directory with timestamp
def create_result_directory():
    """Create a directory with timestamp for storing results"""
    # Create base directory if it doesn't exist
    base_dir = Path("./result")
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_dir / timestamp
    result_dir.mkdir(exist_ok=True)
    
    return result_dir

# Main function
def main():
    # Create result directory
    result_dir = create_result_directory()
    print(f"Results will be saved to {result_dir}")
    
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
    
    # Process data in batches
    batch_size = 50
    all_predictions = []
    
    print(f"Starting batch processing, {batch_size} entries per batch")
    for i in tqdm(range(0, len(dataset), batch_size)):
        # Get current batch texts
        batch_texts = dataset['text'][i:i+batch_size]
        
        # Call API with verification and retry mechanism
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}...")
        batch_predictions = call_api_with_verification(batch_texts, examples)
        
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
        results = evaluate_predictions(all_predictions, dataset['labels'])
        
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