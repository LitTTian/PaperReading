import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    input = input.flatten(0, 1)
    target = target.flatten(0, 1)
        
    sum_dim = (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    return 1 - dice_coeff(input, target, epsilon=epsilon) 