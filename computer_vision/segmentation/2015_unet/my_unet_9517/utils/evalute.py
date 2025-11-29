import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging


def accuracy(model, val_loader, device, mapping): # used for validation
    nb_of_classes = 16
    model.eval()
    n_test = len(val_loader)
    print(n_test)
    accuracy = 0
    total_correct = 0
    total_pixels = 0
    ignore_index = 0
    count_ignore = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, total=n_test, desc='Test round', unit='imgs', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.unsqueeze(0)
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
                mask_pred = model(image)
                mask_pred = mask_pred.argmax(dim=1)
                mask_pred, mask_true = mask_pred.flatten(), mask_true.flatten()
                count_ignore += (mask_true == ignore_index).sum().item() # ignore background
                for cls in range(1, nb_of_classes): # ignore background
                    pred_positive = (mask_pred == cls)
                    actual_positive = (mask_true == cls)
                    true_positive = torch.logical_and(pred_positive, actual_positive).sum().item()
                    total_correct += true_positive
            total_pixels += mask_true.numel()
    accuracy = total_correct / (total_pixels - count_ignore)
    return accuracy

def test(model, test_loader, device, mapping): # used for evaluate checkpoints
    nb_of_classes = 16
    model.eval()
    n_test = len(test_loader)
    print(n_test)
    ious, precisions, recalls = None, None, None
    true_positive = [0] * nb_of_classes
    pred_positive = [0] * nb_of_classes
    actual_positive = [0] * nb_of_classes
    union = [0] * nb_of_classes
    with torch.no_grad():
        for batch in tqdm(test_loader, total=n_test, desc='Test round', unit='imgs', leave=False):
            print(batch['image'].shape)
            image, mask_true = batch['image'], batch['mask']
            image = image.unsqueeze(0)
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            # with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
            autocast_device = 'cpu'
            with torch.autocast(autocast_device):
                mask_pred = model(image)
                mask_pred = mask_pred.argmax(dim=1)
                mask_pred, mask_true = mask_pred.flatten(), mask_true.flatten()
                for cls in range(1, nb_of_classes): # ignore background
                    pred_positive[cls] += (mask_pred == cls).sum().item()
                    actual_positive[cls] += (mask_true == cls).sum().item()
                    true_positive[cls] += torch.logical_and(mask_pred == cls, mask_true == cls).sum().item()
                    union[cls] += torch.logical_or(mask_pred == cls, mask_true == cls).sum().item()
    ious = [true_positive[cls] / union[cls] if union[cls] != 0 else float('nan') for cls in range(nb_of_classes)]
    iou_per_class = {str(mapping[i]): ious[i] for i in range(len(ious))}
    print(true_positive)
    print(pred_positive)
    for cls, iou in iou_per_class.items():
        print(f'Class {cls} IoU: {iou:.4f}')
    print(f'Mean IoU: {np.nanmean(ious):.4f}')