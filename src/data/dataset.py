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
    """
    Custom dataset class for SAR (Synthetic Aperture Radar) and optical image pairs.
    
    Args:
        df (pd.DataFrame): DataFrame containing file paths and other metadata.
        train (bool): Boolean indicating whether the dataset is for training (default: True).
        optical_rgb (bool): Boolean indicating whether optical images are Only B8 Bands or RGB bands format (default: False (B8)).
        dem (bool): Boolean indicating whether Digital Elevation Model (DEM) data is included (default: False).
        mask (bool): Boolean indicating whether water mask data is included (default: False).
        resize (tuple): Tuple specifying the desired size for resizing images (default: None).
        transform (callable): Optional transform to be applied to the images (default: None).
        Augment (callable): Optional augmentation function to be applied to the images (default: None).
        device (str): Device to which the tensors will be moved (default: 'cuda').

    """
    
    def __init__(self, df, train = True , optical_rgb=False,
                  dem=False ,mask= False , resize=None, transform = None, Augment = None, device='cuda'):
        super(SAR2OpticalDataset,self).__init__()

        self.df = df
        self.train = train 
        self.optical = optical_rgb
        self.resize = resize
        self.transform = transform 
        self.augment = Augment
        self.device = device
        self.dem = dem
        self.mask = mask

        if self.train:
            self.df = self.df[self.df['train_test'] == 'train']
        else:
            self.df = self.df[self.df['train_test'] == 'test']

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples in the dataset.
        """

        return len(self.df)
    
    def __getitem__(self, indx):
        """
        Retrieves and processes the SAR and optical images at the given index.
        
        Args:
            indx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing SAR image tensor and optical image tensor.
        """
        sar_image = self.get_sar_image(indx)
        optical_image= self.get_optical_image(indx)

        sar_image, optical_image = self.preprocess(sar_image, optical_image)

        return sar_image.to(self.device) , optical_image.to(self.device)
    
    def get_sar_image(self, indx):

        """
        Retrieves the SAR image at the given index.
        
        Args:
            indx (int): Index of the sample to retrieve.
        
        Returns:
            np.ndarray: SAR image data as a NumPy array.
        """
        
        sar_path = self.df.iloc[indx, 2]
        sar_image = imageio.imread(sar_path)
        if self.dem:
            dem = self.df.iloc[indx, 6]
            dem = imageio.imread(dem)
            sar_image = np.dstack((sar_image, dem))

        return sar_image
    
    def get_optical_image(self, indx):

        """
        Retrieves the optical image at the given index.
        
        Args:
            indx (int): Index of the sample to retrieve.
        
        Returns:
            np.ndarray: Optical image data as a NumPy array.
        """

        if self.optical:
            optical_path = self.df.iloc[indx, 4]
        else:
            optical_path = self.df.iloc[indx, 3]

        optical_image = imageio.imread(optical_path)
        if self.mask:
            mask = self.df.iloc[indx, 5]
            mask = cv2.imread(mask,-1)
            optical_image = np.dstack((optical_image, mask))

        return optical_image

    def preprocess(self, sar_image, optical_image):

        """
        Preprocesses the SAR and optical images.
        
        Args:
            sar_image (np.ndarray): SAR image data as a NumPy array.
            optical_image (np.ndarray): Optical image data as a NumPy array.
        
        Returns:
            tuple: A tuple containing preprocessed SAR image tensor and preprocessed optical image tensor.
        """
        
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

    

