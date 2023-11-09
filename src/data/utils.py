from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
import torch
import albumentations as A
import imageio.v2 as imageio


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
