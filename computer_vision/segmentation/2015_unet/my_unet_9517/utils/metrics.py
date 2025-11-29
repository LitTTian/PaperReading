import torch
import numpy as np

def calculate_metrics(pred, target, n_classes, ignore_index=[0]):
    ious = []
    precisions = []
    recalls = []
    accuracys = []
    pred = pred.flatten()
    target = target.flatten()
    
    num_of_true_positive = 0
    for cls in range(n_classes):
        if cls in ignore_index:
            continue # skip background
        pred_positive = (pred == cls)
        actual_positive = (target == cls)
        true_positive = torch.logical_and(pred_positive, actual_positive).sum().item()
        union = torch.logical_or(pred_positive, actual_positive).sum().item()
        false_positive = pred_positive.sum().item() - true_positive
        false_negative = actual_positive.sum().item() - true_positive
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(true_positive / union)
        if true_positive + false_positive == 0:
            precisions.append(float('nan'))
        else:
            precisions.append(true_positive / (true_positive + false_positive))
        if true_positive + false_negative == 0:
            recalls.append(float('nan'))
        else:
            recalls.append(true_positive / (true_positive + false_negative))
        num_of_true_positive += true_positive
    accuracy = num_of_true_positive / len(pred)
    ious, precisions, recalls = np.array(ious), np.array(precisions), np.array(recalls)
    return accuracy, ious, precisions, recalls