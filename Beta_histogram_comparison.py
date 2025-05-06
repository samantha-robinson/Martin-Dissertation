# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:51:14 2025

@author: saman
"""
#%%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# paths
beta_paths = {
    'AssumeHRF': r"D:\betas_session01.nii",
    'FitHRF': r"D:\beta2\betas_session01b2 (1).nii",
    'GLMDenoise': r"D:\beta3\betas_session01b3(2).nii"
}

roi_mask_path = r"D:\Kastner2015.nii\Kastner2015.nii"  

# load ROI mask 
mask_img = nib.load(roi_mask_path)
mask_data = mask_img.get_fdata().astype(bool)
#%%
# prepare and plot histogram for each GLM 
for name, path in beta_paths.items():
    img = nib.load(path)
    data = np.stack([img.dataobj[..., i] for i in range(10)], axis=-1)  # first 10 betas
    roi_betas = data[mask_data]
    values = roi_betas.flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, color='green', edgecolor='black')
    plt.title(f'{name} - Beta Value Distribution (Occipital ROI, First 10 Betas)')
    plt.xlabel('Beta Value')
    plt.ylabel('Voxel Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

