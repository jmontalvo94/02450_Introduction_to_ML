# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:05:50 2019

@author: Emil Chrisander
"""

import pandas as pd
import numpy as np

## Set URL for raw data
url_raw_data = "https://drive.google.com/uc?export=download&id=1DEmiVdf5UGOo8lqNvNiiHqJQuiNCgxc2"
#url_raw_data = "http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data"


## Load raw data file (South African Heart Disease dataset)
df_heart_disease = pd.read_csv(url_raw_data)

## Initial data manipulations
df_heart_disease = df_heart_disease.drop(columns = "row.names", axis = 1) ## erases column 'row.names'. Axis = 1 indicates it is a column rather than row drop.

df_heart_disease.head()