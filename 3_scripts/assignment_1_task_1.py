# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:05:50 2019

@author: Emil Chrisander
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd

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
print("Show content of dataframe")
print(df_heart_disease.head())

#######################################################
### STANDARDIZATION OF ATTRIBUTES #####################
#######################################################

## Start by creating a matric representation of the dataframe (only keep attributes)
X = df_heart_disease.drop(columns = ["chd","chd_cat","famhist"], axis = 1).to_numpy(dtype=np.float32) ## Type is set to float to allow for math calculations

## Store dimensions of X as local variables
N = np.shape(X)[0] ## Number of observations
M = np.shape(X)[1] ## Number of attributes

## Substract mean values from X (create X_tilde)
X_tilde = X - np.ones((N,1))*X.mean(axis=0)

## Divide by std. deviation from X_tilde
X_tilde = X_tilde*(1/np.std(X_tilde,0))

## Set attribute names
attribute_names = df_heart_disease.drop(columns = ["chd","chd_cat","famhist"], axis = 1).columns

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
print(round(df_heart_disease.describe(),2))

print("Summary statistics (categorial variables)")
print(round(df_heart_disease.describe(include='category'),0))

#######################################################
### OUTLIER DETECTION  ################################
#######################################################
plt.boxplot(X_tilde)
plt.xticks(range(1,M+1),attribute_names, rotation=90)
plt.ylabel('Standardized value')
plt.title('African Heart Disease Data - boxplot')
plt.show()


#######################################################
### CONDITIONAL BOX PLOTS ON CHD  #####################
#######################################################

## Create a grid of subplots with box plots conditional on chd
grid_plt = plt.figure(figsize=(10,10),dpi=90)
grid_plt.subplots_adjust(hspace=0.4, wspace=0.4)
grid_plt.suptitle('Distribution of features conditional on outcome variable')
for i,feature in zip(range(1,M+1),attribute_names):
    ax = grid_plt.add_subplot(3,3,i)
    if(feature!="famhist_present"):
        #sns.catplot(y="chd_cat",x=feature, data=df_heart_disease, kind = "swarm",ax=ax); ## Swamp plot instead
        sns.catplot(y="chd_cat",x=feature, data=df_heart_disease, kind = "box",ax=ax);
        ax.set_ylabel("Diagnosed with heart disease", fontsize=10)
    else:
        sns.countplot(x=feature, data=df_heart_disease,hue="chd_cat",ax=ax)
        ax.legend().set_visible(False) ## Do not show legend
        ax.set_ylabel("# of obs.", fontsize=10)
    ax.set_xlabel(feature, fontsize=12)
    ax.tick_params(axis='both',labelsize=10)

## Save the grid plot    
grid_plt.savefig('individual_box_plot_conditional_on_chd.png')



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
### PCA ANALYSIS ######################################
#######################################################

# PCA by computing SVD of X_tilde
U,S,V = svd(X_tilde,full_matrices=False)

## Calculate rho
rho = (S*S) / (S*S).sum() 



print("First PCA")
print(V[0])

## Set threshold for variance explained
threshold = 0.95

# Choose two PCs to plot (the projection)
i = 0
j = 1

# Plot variance explained
plt.figure(figsize=(5,5),dpi=90)
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

## Project of eigen values on our feature matrix (X_tilde)
Z = U*S;

## Plot projects of PC 1 and PC 2 to X_tilde
project_plot = sns.scatterplot(Z[:,i],Z[:,j], hue = df_heart_disease.chd_cat)
project_plot.set_xlabel("PC "+str(i+1))
project_plot.set_ylabel("PC "+str(j+1))
project_plot.set_title('Projection of PCs')

## Plot direction of attribute coefficients
plt.figure(figsize=(5,5),dpi=90)
for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(V[att,i], V[att,j], attribute_names[att],fontsize=10,bbox=dict(facecolor='blue', alpha=0.5))
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
     np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Direction of attribute coefficients')
plt.axis('equal')
