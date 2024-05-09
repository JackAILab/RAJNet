import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from scipy import ndimage
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py

def scipy_rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
    volume[volume < 0] = 0 
    volume[volume > 1] = 1
    return volume


class BasicDataset(Dataset):
    def __init__(self, image_path, label,data_type,k_threshold,transform=None):
        # 240420
        self.train_save_data_path = './Data/h5f_data/train_small_data.he5'   
        self.train_save_label_path = './Data/h5f_data/train_small_data_label.he5'

        # 测试保存路径
        self.test_save_data_path = './Data/h5f_data/test_small_data.he5'
        self.test_save_label_path = './Data/h5f_data/test_small_data_label.he5'


        self.image_path = image_path
        self.label = label
        self.transform = transform
        self.data_type=data_type

        if self.data_type=='train':
            save_data_path=self.train_save_data_path
            save_label_path=self.train_save_label_path 

        elif self.data_type=='test':
            save_data_path=self.test_save_data_path
            save_label_path=self.test_save_label_path
        
        data_h5f = h5py.File(save_data_path , 'r')
        label_h5f = h5py.File(save_label_path, 'r')

        self.length = len(data_h5f)  

        self.keys = list(data_h5f.keys()) 
        random.shuffle(self.keys)
        data_h5f.close()
        label_h5f.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.data_type=='train':
            global save_data_path
            global save_label_path
            save_data_path=self.train_save_data_path
            save_label_path=self.train_save_label_path 

        elif self.data_type=='test':
            save_data_path=self.test_save_data_path
            save_label_path=self.test_save_label_path

        data_h5f = h5py.File(save_data_path, 'r')
        label_h5f = h5py.File(save_label_path, 'r')

        key = self.keys[idx]
        
        data_arr = np.array(data_h5f[key]) # 将数据转化为矩阵
        label_arr = np.array(label_h5f[key])
        
        
        data_h5f.close()
        label_h5f.close()     
        
        return data_arr, label_arr 



