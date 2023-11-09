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

def generate_patches_with_overlap(image, patch_size=512, overlap=0.5):

    """
    Generate patches from an image with specified patch size and overlap.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        patch_size (int): Size of the patches (default: 512).
        overlap (float): Overlap ratio between patches (default: 0.5).

    Returns:
        np.ndarray: Array containing generated patches.
    """

    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width, channels = image.shape

    stride = int(patch_size * (1 - overlap))

    patches = []
    for h in range(0, height  - stride + 1, stride):
        h_min = h
        h_max = h_min + patch_size
        if h_max > height:
            h_max = height
            h_min = h_max - 512
        for w in range(0, width - stride + 1, stride):
            w_min = w
            w_max = w_min + patch_size
            if w_max > width:
                w_max = width
                w_min = w_max - 512
            patch = image[h_min:h_max, w_min:w_max]
            patches.append(patch)

    return np.array(patches)


def stitch_patches(patches, original_height, original_width, patch_size=512, stride=256):
    """
    Stitch patches together using weighted averaging blending technique.

    Args:
        patches (np.ndarray): Array of patches to be stitched.
        original_height (int): Original height of the image.
        original_width (int): Original width of the image.
        patch_size (int): Size of the patches (default: 512).
        stride (int): Stride for overlapping patches (default: 256).

    Returns:
        np.ndarray: Stitched image as a NumPy array.
    """

    stitched_image = np.zeros((original_height, original_width, 1))
    count = np.ones((original_height, original_width, 1))
    i = 0

    for h in range(0, original_height - stride + 1, stride):
        h_min = h
        h_max = h_min + patch_size
        if h_max > original_height:
            h_max = original_height
            h_min = h_max - 512
        for w in range(0, original_width - stride + 1, stride):
            w_min = w
            w_max = w_min + patch_size
            if w_max > original_width:
                w_max = original_width
                w_min = w_max - 512
            stitched_image[h_min:h_max, w_min:w_max] = weighted_average_blending(
                stitched_image[h_min:h_max, w_min:w_max], patches[i])
            count[h_min:h_max, w_min:w_max] += 1
            i += 1

    count[count == 0] = 1
    stitched_image /= count

    return stitched_image


def weighted_average_blending(patch1, patch2):
    """
    Perform weighted averaging blending between two overlapping patches.

    Args:
        patch1 (np.ndarray): First input patch as a NumPy array.
        patch2 (np.ndarray): Second input patch as a NumPy array.

    Returns:
        np.ndarray: Blended patch as a NumPy array.
    """
    alpha = 0.5  # Weight factor for blending, you can adjust this value as needed
    blended_patch = alpha * patch1 + (1 - alpha) * patch2
    return blended_patch
