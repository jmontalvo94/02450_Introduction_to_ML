# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:05:50 2019

@author: Emil Chrisander
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

## Set URL for raw data
url_raw_data = "https://drive.google.com/uc?export=download&id=1DEmiVdf5UGOo8lqNvNiiHqJQuiNCgxc2" ## This is a download url for a text file located on google drive. The original dataset is only available as a .rdata file (r data file). Thus, we have downloaded a text version and uploaded to a google drive server instead. 


## Load raw data file (South African Heart Disease dataset)
df_heart_disease = pd.read_csv(url_raw_data)

## Initial data manipulations
df_heart_disease = df_heart_disease.drop(columns = "row.names", axis = 1) ## erases column 'row.names'. Axis = 1 indicates it is a column rather than row drop.
df_heart_disease["famhist"] = df_heart_disease["famhist"].astype('category') ## Set discrete variables to categorial type
df_heart_disease["chd_cat"] = df_heart_disease["chd"].astype('category') ## Set discrete variables to categorial type
df_heart_disease["famhist_present"] = pd.get_dummies(df_heart_disease.famhist, prefix='famhist_',drop_first=True)
df_heart_disease = df_heart_disease.drop(columns = "famhist", axis = 1) ## erases column 'row.names'. Axis = 1 indicates it is a column rather than row drop.


## Show content of dataframe
print("Show content of dataframe")
print(df_heart_disease.head())


#######################################################
### MISSING DATA ANALYSIS #############################
#######################################################

## Check for NA observations
print("Count of NA observations (0 means no NA in variable)")
print(df_heart_disease.isnull().sum())


#######################################################
### DESCRIPTIVE SUMMARY STATISTICS ####################
#######################################################

print("Summary statistics (numerical variables)")
print(round(df_heart_disease.describe(),0))

print("Summary statistics (categorial variables)")
print(round(df_heart_disease.describe(include='category'),0))

#######################################################
### OUTLIER DETECTION  ################################
#######################################################


#######################################################
### DISTRIBUTION OF VARIABLES##########################
#######################################################

print("Histogram of attributes")
print(df_heart_disease.hist())

#######################################################
### CORRELATION TABLE #################################
#######################################################

print("Correlation Matrix")
print(round(df_heart_disease.corr(method='pearson'),1))

#######################################################
### STANDARDIZATION OF ATTRIBUTES #####################
#######################################################

## Start by creating a matric representation of the dataframe (only keep attributes)
X = df_heart_disease.drop(columns = "chd", axis = 1).to_numpy(dtype=np.float32) ## Type is set to float to allow for math calculations

## Store dimensions of X as local variables
N = np.shape(X)[0] ## Number of observations
M = np.shape(X)[1] ## Number of attributes

## Substract mean values from X (create X_tilde)
X_tilde = X - np.ones((N,1))*X.mean(axis=0)

## Substract std. deviation from X_tilde
X_tilde = X_tilde*(1/np.std(X_tilde,0))

# PCA by computing SVD of X_tilde
U,S,V = svd(X_tilde,full_matrices=False)

## Calculate rho
rho = (S*S) / (S*S).sum() 

