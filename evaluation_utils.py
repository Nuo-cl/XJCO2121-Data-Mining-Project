import numpy as np
import evaluate
from sklearn.metrics import precision_recall_fscore_support

# Calculate IoU (Intersection over Union) for each class
def calculate_iou(pred_labels, true_labels, num_classes, id_to_emotion):
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
def evaluate_predictions(predictions, references, id_to_emotion):
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
    iou_scores = calculate_iou(
        predictions, 
        references, 
        num_classes=max(unique_labels) + 1,
        id_to_emotion=id_to_emotion
    )
    
    # Filter IoU scores to only include emotions that appear in the data
    filtered_iou_scores = {}
    for label in unique_labels:
        emotion_name = id_to_emotion.get(label, f"Unknown-{label}")
        if emotion_name in iou_scores:
            filtered_iou_scores[emotion_name] = iou_scores[emotion_name]
    # Add mean IoU
    filtered_iou_scores["mean_iou"] = float(np.mean(list(filtered_iou_scores.values())))
    results["iou_scores"] = filtered_iou_scores
    
    # Add metadata about the evaluation
    results["metadata"] = {
        "num_unique_labels": len(unique_labels),
        "labels_present": unique_labels,
        "total_samples": len(predictions)
    }
    
    return results 