# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:16:35 2025

@author: saman
"""
#%% load in files 

import numpy as np
import pandas as pd
import statsmodels.api as sm

# load time series 
voxel_time_series = np.load(r"C:\Users\saman\OneDrive\Desktop\year_3\3rd Year Dissertation Project\time_series.npy") 

# load the design matrix
design_matrix = np.load(r"C:/Users/saman/OneDrive/Desktop/year_3/3rd Year Dissertation Project/January 2025/design_matrix.npy")  

#%% fit GLM 

# add an intercept to the design matrix
design_matrix = sm.add_constant(design_matrix)  

# fit the GLM
model = sm.OLS(voxel_time_series, design_matrix).fit()

# extract beta weights
beta_weights = model.params

# print results
print("Beta weights:", beta_weights)
print("p-values:", model.pvalues)
print("R-squared:", model.rsquared)
