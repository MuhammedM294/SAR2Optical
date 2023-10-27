from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
import torch
import albumentations as A
import imageio.v2 as imageio


class SAR2OpticalDataset(Dataset):
    
    def __init__(self, df, train = True , optical_rgb=False,
                  dem=False ,resize=None, transform = None, Augment = None, device='cuda'):
        super(SAR2OpticalDataset,self).__init__()

        self.df = df
        self.train = train 
        self.optical = optical_rgb
        self.resize = resize
        self.transform = transform 
        self.augment = Augment
        self.device = device
        self.dem = dem

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, indx):
        sar_image = self.get_sar_image(indx)
        optical_image= self.get_optical_image(indx)

        sar_image, optical_image = self.preprocess(sar_image, optical_image)

        return sar_image.to(self.device) , optical_image.to(self.device)
    
    def get_sar_image(self, indx):

        sar_path = self.df.iloc[indx, 3]
        sar_image = imageio.imread(sar_path+".tif")

        return sar_image
    
    def get_optical_image(self, indx):

        if self.optical:
            optical_path = self.df.iloc[indx, 5]
        else:
            optical_path = self.df.iloc[indx, 4]

        optical_image = imageio.imread(optical_path+".tif")

        return optical_image

    def preprocess(self, sar_image, optical_image):
        
        if len(sar_image.shape) == 2:
            sar_image = np.expand_dims(sar_image, axis=-1)
        if len(optical_image.shape) == 2:
            optical_image = np.expand_dims(optical_image, axis=-1)
        
        if self.augment:
            data =self.augment()(image = sar_image,mask = optical_image)
            sar_image = data['image']
            optical_image = data['mask']
        
        sar_image = torch.Tensor(sar_image).float().permute(2,0,1)
        optical_image = torch.Tensor(optical_image ).float().permute(2,0,1)

        return sar_image, optical_image

    

def train_augmentation():
    """
        Augmentation pipeline for training data.

        Returns:
            A.Compose: Augmentation pipeline object.
    """

    return  A.Compose([ 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                            border_mode=cv2.BORDER_REFLECT),
        
    ])


