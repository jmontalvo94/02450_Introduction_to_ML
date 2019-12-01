# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:26:57 2019
@author: Emil Chrisander, Julian Böhm, and Jorge M. Arvizu
"""

import pandas as pd
import numpy as np
from apyori import apriori

## Set URL for raw data
url_raw_data = "https://drive.google.com/uc?export=download&id=1DEmiVdf5UGOo8lqNvNiiHqJQuiNCgxc2" ## This is a download url for a text file located on google drive. The original dataset is only available as a .rdata file (r data file). Thus, we have downloaded a text version and uploaded to a google drive server instead. 

## Load raw data file (South African Heart Disease dataset)
df_heart_disease = pd.read_csv(url_raw_data)

## Initial data manipulations
df_heart_disease = df_heart_disease.drop(columns = "row.names", axis = 1) ## erases column 'row.names'. Axis = 1 indicates it is a column rather than row drop.
#df_heart_disease["famhist"] = df_heart_disease["famhist"].astype('category') ## Set discrete variables to categorial type
df_heart_disease["chd_cat"] = df_heart_disease["chd"].astype('category').cat.rename_categories(["No","Yes"]) ## Set discrete variables to categorial type with No and Yes.
df_heart_disease["famhist_present"] = pd.get_dummies(df_heart_disease.famhist, prefix='famhist_',drop_first=True)
#df_heart_disease = df_heart_disease.drop(columns = "famhist", axis = 1) ## erases column 'row.names'. Axis = 1 indicates it is a column rather than row drop.


## Show content of dataframe
#print("Show content of dataframe")
#print(df_heart_disease.head())

## Define attribute names
attribute_names          = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age','famhist_present']
class_name               = ['chd']

#######################################################
### PROBLEM A: CLUSTER ANALYSIS     ###################
#######################################################


#######################################################
### PROBLEM B: OUTLIER DETECTION    ###################
#######################################################
## Start by creating a matric representation of the dataframe (only keep attributes)
X = df_heart_disease[attribute_names].to_numpy(dtype=np.float32) ## Type is set to float to allow for math calculations
y = df_heart_disease[class_name].to_numpy(dtype=np.float32)

## Store dimensions of X as local variables
N = np.shape(X)[0] ## Number of observations
M = np.shape(X)[1] ## Number of attributes
#normalization
## Substract mean values from X (create X_tilde)
X_tilde = X - np.ones((N,1))*X.mean(axis=0)

## Divide by std. deviation from X_tilde
X_tilde = X_tilde*(1/np.std(X_tilde,0))

## Overwrite X_tilde as X, since we won't be using the non-standardized X
X = X_tilde

from matplotlib.pyplot import figure, bar, title, show, plot
from scipy.stats.kde import gaussian_kde
from toolbox_02450 import gausKernelDensity

###################  KDE   ############################
#######################################################

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i_kde,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i_kde,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i_kde] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i_kde = (density.argsort(axis=0)).ravel()
density = density[i_kde].reshape(-1,)

# Plot density estimate of outlier score
figure(1)
bar(range(20),density[:20])
title('Gausian Kernel Density: Outlier score')

#########  KNN density estimation   ###################
#######################################################
from sklearn.neighbors import NearestNeighbors
### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i_den = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i_den = density.argsort()
density = density[i_den]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20),density[:20])
title('KNN density: Outlier score')

##  KNN average relative density estimation   #########
#######################################################

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(20),avg_rel_density[:20])
title('KNN average relative density: Outlier score')

### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:,K-1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

# Plot k-neighbor estimate of outlier score (distances)
figure(7)
bar(range(20),score[:20])
title('5th neighbor distance: Outlier score')
    
#######################################################
### PROBLEM C: ASSOCIATION MINING   ###################
#######################################################

#numeric_attribute_names      = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']
#categorical_attributes_names = ['famhist_present','chd'] 
#
### Set control parameters
#numeric_var_splits = 4 # This var determines the number of splits for binarization of the numeric variable. 2 corresponds to a 50:50 split, 4 to 25:25:25:25 split, etc. 
#
### Define helping functions
#def mat2transactions(X, labels=[]):
#    T = []
#    for i in range(X.shape[0]):
#        l = np.nonzero(X[i, :])[0].tolist()
#        if labels:
#            l = [labels[i] for i in l]
#        T.append(l)
#    return T
#
#def apriori_rules_as_df(rules,y_itemsets):
#    frules = []
#    for r in rules:
#        for o in r.ordered_statistics:
#            if(len(set(y_itemsets).intersection(list( o.items_add)))>0):
#                conf = round(o.confidence,ndigits=2)
#                supp = round(r.support,ndigits=2)
#                x = ", ".join( list( o.items_base ) )
#                y = ", ".join( list( o.items_add ) )
#                #print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
#                frules.append( (x,y,supp,conf) )
#    df_frules = pd.DataFrame(frules,columns=['X','Y','support','confidence'])\
#    .sort_values(by='support', ascending = False)\
#    .assign(lift = lambda df:round(df.confidence / df.support,ndigits=1))\
#    .reset_index().drop(columns='index')            
#    return df_frules
#
## create quantile arrays
#quantiles       = np.linspace(0,1,numeric_var_splits+1)
#quantiles_lower = quantiles[0:numeric_var_splits]
#quantiles_upper = quantiles[1:]
#
## Run loop to create binarized data frame
#df_binarized = pd.DataFrame()
#for var in numeric_attribute_names:
#    #var = 'sbp'
#    df_tmp = df_heart_disease[[var]].assign(tmp_var = "")
#    for quantile_lower,quantile_upper in zip(quantiles_lower,quantiles_upper):
#        low_quantile_value  = df_tmp[[var]].quantile(quantile_lower).values[0]
#        high_quantile_value = df_tmp[[var]].quantile(quantile_upper).values[0]
#        df_tmp.loc[(df_tmp[var]>=low_quantile_value) & (df_tmp[var]<high_quantile_value),'tmp_var'] = str(int(quantile_lower*100)) + "%_to_" + str(int(quantile_upper*100)) + "%"
# 
#    df_tmp = pd.get_dummies(df_tmp['tmp_var'], prefix=var,drop_first=True)
#    df_binarized = pd.concat([df_binarized,df_tmp],axis = 1)
#
#for var in categorical_attributes_names:
#    df_binarized = pd.concat([df_binarized,df_heart_disease[[var]]], axis = 1)
#    df_binarized = df_binarized.assign(tmp_var = lambda df:1 - df[var])
#    df_binarized = df_binarized.rename(columns={var:var+"_yes",'tmp_var':var+"_no"})
#    
#X = df_binarized.to_numpy(dtype=np.int32)
#
#
## Transform binarized data to label values
#T = mat2transactions(X, labels = list(df_binarized.columns))
#
## find association rules with apriory algorithm
#rules = apriori(T, min_support=0.05, min_confidence=.6)
#y_itemsets   = ['chd_yes']
#df_chd_rules = apriori_rules_as_df(rules,y_itemsets)
#
### Save df table as latex format
#with pd.option_context("max_colwidth",1000):
#    latex_table  = df_chd_rules.to_latex(index=False, column_format='p{7.5cm}rrrr')
#with open('association_rules_table.tex','w') as tf:
#    tf.write(latex_table)