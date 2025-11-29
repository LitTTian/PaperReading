- Modify the dataset path:
```python
dataset_dir = Path('../../data/processed')
# ori path
dir_img = dataset_dir / 'train/image'
dir_mask = dataset_dir / 'train/indexLabel'

val_dir_img = dataset_dir / 'val/image'
val_dir_mask = dataset_dir / 'val/indexLabel'

dir_checkpoint = Path('./checkpoints/')
```

- About args:
```python
arg_dict = {
    'epochs': 30, # epoch to train
    'batch_size': 4, # input batch size for training
    'lr': 1e-5, # initial learning rate
    'load': None, # './checkpoints/bs4rs0.5_new/checkpoint_epoch20_continue.pth',
                  # we save optimizer and model state in case of crash
                  # so that we can continue training by modifying this parameter
    'scale': 0.5, # downscaling factor of the images
    'val': 10.0, # percent of the data that is used as validation (0-100)
    'num_classes': 16 # number of classes in the dataset
}
```