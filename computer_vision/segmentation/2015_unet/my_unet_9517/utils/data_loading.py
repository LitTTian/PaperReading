import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.io import read_image
from torchvision import transforms

def unique_mask_values(mask_file): # 获取mask中所有的类别
    mask = np.asarray(Image.open(mask_file))
    return np.unique(mask)

class MergeClass(torch.nn.Module): # 合并类别
    def __init__(self, map):
        super(MergeClass, self).__init__()
        self.class_map = map

    def forward(self, img):
        new_img = self.class_map[img]
        return new_img


class ImageResize(torch.nn.Module):
    def __init__(self, new_size, interpolate_mode=Image.NEAREST):
        super(ImageResize, self).__init__()
        self.new_size = new_size
        self.interpolate_mode = interpolate_mode # Image.NEAREST if is_mask else Image.BICUBIC

    def forward(self, img):
        img = img.resize(self.new_size, resample=self.interpolate_mode)
        img = np.asarray(img)
        return img


class ImageNormalization(torch.nn.Module):
    def __init__(self):
        super(ImageNormalization, self).__init__()
        
    def forward(self, img):
        img = img / 255.0
        return img
        

class WSDataset(Dataset):
    # 2016x 1512
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, im_dir, mask_dir, scale, nb_of_classes, merge_classes=None, dataset_analyse=False):
        self.im_dir = im_dir
        self.mask_dir = mask_dir
        self.img_names = list(set(filename for filename in listdir(im_dir) if filename.endswith('png')) & 
                              set(filename for filename in listdir(mask_dir) if filename.endswith('png')))
        
        img_shape = read_image(join(self.im_dir, self.img_names[0])).shape
        self.new_Size = int(img_shape[1] * scale), int(img_shape[2] * scale)
        self.map = self.compute_map(nb_of_classes + 1, merge_classes) # 从原始标签类别到最终训练类别的映射

        self.im_transform = transforms.Compose([
                            ImageResize(self.new_Size, interpolate_mode=Image.BICUBIC), 
                            ImageNormalization(),
                            transforms.ToTensor()
                        ])
        self.mask_transform =  transforms.Compose([
                            ImageResize(self.new_Size, interpolate_mode=Image.NEAREST),
                            MergeClass(self.map),
                            # transforms.ConvertImageDtype(torch.long)
                        ])
        
        self.mask_values = nb_of_classes # 16
        if dataset_analyse:
            self.dataset_analyse()

    def __getitem__(self, idx):
        im_path = join(self.im_dir, self.img_names[idx])
        im = Image.open(im_path)
        im = self.im_transform(im)

        mask_path = join(self.mask_dir, self.img_names[idx])
        mask = Image.open(mask_path)
        mask = self.mask_transform(mask)
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        # return im, mask
        return {
            'image': im,
            'mask': mask
        }
    
    def __len__(self):
        return len(self.img_names)
    
    def compute_map(self, nb_of_classes, merge_dict): # 构建类别映射
        nb_of_classes += len(merge_dict)
        map_list = [i for i in range(nb_of_classes)]
        shift = 0
        for i in range(nb_of_classes):
            if i in merge_dict:
                shift += 1
            map_list[i] -= shift
        for k, v in merge_dict.items():
            map_list[k] = map_list[v]
        return np.array(map_list)
    
    def dataset_analyse(self):
        logging.info(f'Creating dataset with {len(self.img_names)} examples')
        logging.info('Scanning mask files to determine unique values')
        mask_files = [join(self.mask_dir, img_name) for img_name in self.img_names]
        
        with Pool() as p: # 多进程读取mask文件
            unique_values = list(tqdm(
                p.imap(unique_mask_values, mask_files),
                total=len(mask_files)
            ))
        
        unique = np.unique(np.concatenate(unique_values))
        self.mask_values = self.map[unique].tolist()

        logging.info(f'Unique mask values: {self.mask_values}')


