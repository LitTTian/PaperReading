import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

import wandb
from evaluate import accuracy
from unet.unet_model import UNet
from utils.data_loading import WSDataset 
from utils.dice_score import dice_loss

import time


dataset_dir = Path('../../data/processed')
# ori path
dir_img = dataset_dir / 'train/image'
dir_mask = dataset_dir / 'train/indexLabel'

val_dir_img = dataset_dir / 'val/image'
val_dir_mask = dataset_dir / 'val/indexLabel'

dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 10,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        optimizer_state_dict: dict = None,
        current_epoch: int = 0,
        merge_classes: dict = None,
        checkpoint_name: str = None
):
    train_set = WSDataset(dir_img, dir_mask, scale=img_scale, nb_of_classes=model.n_classes, merge_classes=merge_classes, dataset_analyse=False)
    val_set = WSDataset(val_dir_img, val_dir_mask, scale=img_scale, nb_of_classes=model.n_classes, merge_classes=merge_classes, dataset_analyse=True)

    n_train = len(train_set)
    n_val = len(val_set)

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        logging.info('Optimizer state loaded from checkpoint')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() # cross entropy loss
    # number of pixels for each class
    pixel_counts = torch.tensor([118755, 1600840094, 6857813, 48116742, 103405446, 862782, 2639454701, 10507673624, 203633644, 1620768, 60550301, 14315444, 65429710, 19023004, 1350206976, 1822499988])
    pixel_counts = pixel_counts[1:] # remove background
    # compuyte weights
    # weights = 1 / torch.tensor(pixel_counts) # use inverse of pixel counts as weights, is not good
                                               # because some classes have extremely large or small pixel counts
    total_pixel_count = sum(pixel_counts)
    epsilon = 1e-6
    log_weights = torch.log(total_pixel_count/ (pixel_counts + epsilon))

    weights = log_weights
    weights = weights / weights.sum()  # normalize
    weights = torch.cat((torch.tensor([0.0]), weights)) # 所有分类的权重
    weights = weights.to(device)
    def weighted_ce_loss(pred, target): # weighted-cross entropy loss
        return F.cross_entropy(pred, target, weight=weights)
    global_step = 0

    for epoch in range(current_epoch+1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # HL: 损失函数
                    loss = weighted_ce_loss(masks_pred, true_masks)  # MODULE: 加权交叉熵损失函数
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()


                # Evaluation round - 验证阶段
                division_step = (n_train // (5 * batch_size))
                # division_step = (n_train // (600 * batch_size)) # for debug
                if division_step > 0:
                    if global_step % division_step == 0:

                        # accuracy, mIou, precision, recall = evaluate(model, val_loader, device, amp)
                        acc = accuracy(model, val_loader, device, amp)
                        scheduler.step(acc)

                        logging.info(f'Validation round:')
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Accuracy': acc,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                            })
                        except:
                            pass

        if save_checkpoint:
            if checkpoint_name is None:
                checkpoint_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) # use time as name

            dir_checkpoint = Path('./checkpoints/')
            if checkpoint_name is not None:
                dir_checkpoint = dir_checkpoint / checkpoint_name # new path
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True) # 创建文件夹

            # save optimizer state
            checkpoint = {
                'model_state_dict': model.state_dict(), # model state
                'optimizer_state_dict': optimizer.state_dict(), # optimizer state
                'loss': loss, # loss
                'epoch': epoch, # epoch
                'mask_values': val_set.mask_values # mask
            }
            torch.save(checkpoint, str(dir_checkpoint / 'checkpoint_epoch{}_continue.pth'.format(epoch))) 
            logging.info(f'Checkpoint continue version {epoch} saved!')

            state_dict = model.state_dict()
            state_dict['mask_values'] = val_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!') 
        
if __name__ == '__main__':
    arg_dict = {
        'epochs': 30,
        'batch_size': 4,
        'lr': 1e-5,
        'load': None, # './checkpoints/bs4rs0.5_new/checkpoint_epoch20_continue.pth',
        'scale': 0.5,
        'val': 10.0,
        'amp': False,
        'num_classes': 16
    }
    parser = argparse.ArgumentParser()
    for key in arg_dict:
        parser.add_argument(f'--{key}', default=arg_dict[key], type=type(arg_dict[key]))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # define the merge classes in the dataset
    merge_classes = {
        1: 6, # asphalt -> other-terrain
        12: 16, # pole -> other-object
        13: 0, # exclude from evaluation
    }

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.num_classes)
    model = model.to(memory_format=torch.channels_last)

    optimizer_state_dict = None
    current_epoch = 0

    if args.load: # if load checkpoint
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f'Model loaded from {args.load}')
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        current_epoch = checkpoint['epoch']

    model.to(device=device)
    try:
        torch.cuda.empty_cache()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            optimizer_state_dict=optimizer_state_dict,
            current_epoch=current_epoch,
            merge_classes=merge_classes
        )
    except torch.cuda.OutOfMemoryError:
        print('OOM error')