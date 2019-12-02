# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:26:57 2019

@author: Emil Chrisander, Julian Böhm, and Jorge Montalvo Arvizu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
from toolbox_02450 import clusterplot, clusterval
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.linalg import svd
from sklearn.mixture import GaussianMixture
from sklearn import model_selection

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


#%%
#######################################################
### PROBLEM A: CLUSTER ANALYSIS     ###################
#######################################################

## Start by creating a matric representation of the dataframe (only keep attributes)
X = df_heart_disease[attribute_names].to_numpy(dtype=np.float32) ## Type is set to float to allow for math calculations
y = df_heart_disease[class_name].to_numpy(dtype=np.float32).squeeze()
# X = np.delete(X,-1,axis=1)

oneK = True

if (oneK):
    ## One-out-of-K encoding of famhist_present
    famhist = np.array(X[:, -1], dtype=int).T
    K = famhist.max()+1
    famhist_encoding = np.zeros((famhist.size, K))
    famhist_encoding[np.arange(famhist.size), famhist] = 1
    X = np.concatenate( (X[:, :-1], famhist_encoding), axis=1) 

## Store dimensions of X as local variables
N = np.shape(X)[0] ## Number of observations
M = np.shape(X)[1] ## Number of attributes

## Substract mean values from X (create X_tilde)
X_tilde = X - np.ones((N,1))*X.mean(axis=0)

## Divide by std. deviation from X_tilde
X_tilde = X_tilde*(1/np.std(X_tilde,0))

## Overwrite X_tilde as X, since we won't be using the non-standardized X
X = X_tilde

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'ward'
Metric = 'euclidean'

Methods = ['single', 'complete', 'average', 'ward']

# Initiate validities vectors:
Rand = np.zeros((len(Methods),))
Jaccard = np.zeros((len(Methods),))
NMI = np.zeros((len(Methods),))

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
plt.figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)
unique, counts = np.unique(cls, return_counts=True)

# Display dendrogram
max_display_levels=2
plt.figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
plt.show()

# Calculate validities
for m in range(len(Methods)):
    # run hierarchical clustering:
    Z = linkage(X, method=Methods[m], metric=Metric)
    Maxclust = 2
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)
    # compute cluster validities:
    Rand[m], Jaccard[m], NMI[m] = clusterval(y,cls)
    
        
# Plot results:

plt.figure(5)
plt.title('Cluster validity')
plt.plot(np.arange(len(Methods))+1, Rand)
plt.plot(np.arange(len(Methods))+1, Jaccard)
plt.plot(np.arange(len(Methods))+1, NMI)
plt.legend(['Rand', 'Jaccard', 'NMI'], loc=4)
plt.show()


# PCA

# PCA by computing SVD of X_tilde
U,S,V = svd(X,full_matrices=False)

## Calculate rho
rho = (S*S) / (S*S).sum() 

print("First PCA")
print(V[0])
print("Second PCA")
print(V[1])

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
project_plot = sns.scatterplot(Z[:,i],Z[:,j], hue = df_heart_disease.chd_cat, style = cls)
project_plot.set_xlabel("PC "+str(i+1))
project_plot.set_ylabel("PC "+str(j+1))
project_plot.set_title('Projection of PCs')

# Gaussian Mixture Model

# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'       # 'full' or 'diag'
reps = 10                 # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        
        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results

plt.figure(7); 
plt.plot(KRange, BIC,'-*b')
plt.plot(KRange, AIC,'-xr')
plt.plot(KRange, 2*CVE,'-ok')
plt.legend(['BIC', 'AIC', 'Crossvalidation'])
plt.xlabel('K')
plt.savefig('CV.png')
plt.show()


# When K is selected

# Number of clusters
K = 2
cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'kmeans'

# number of fits with different initalizations, best result will be kept
reps = 10

# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)

# extract cluster labels
cls = gmm.predict(X)    

# extract cluster centroids (means of gaussians)
cds = gmm.means_        

# extract cluster shapes (covariances of gaussians)
covs = gmm.covariances_

if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
plt.figure(figsize=(14,9))
idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
plt.savefig('K2.png')
plt.show()

## Plot projects of PC 1 and PC 2 to X_tilde
project_plot = sns.scatterplot(Z[:,i],Z[:,j], hue = df_heart_disease.chd_cat, style = cls)
project_plot.set_xlabel("PC "+str(i+1))
project_plot.set_ylabel("PC "+str(j+1))
project_plot.set_title('Projection of PCs')

RandGMM = np.zeros((3,))
JaccardGMM = np.zeros((3,))
NMIGMM = np.zeros((3,))

#compute cluster validities:
unique, counts = np.unique(cls, return_counts=True)
RandGMM[0], JaccardGMM[0], NMIGMM[0] = clusterval(y,cls)

# run hierarchical clustering and compare with GMM:
Z = linkage(X, method='ward', metric=Metric)
Maxclust = 2
clsH = fcluster(Z, criterion='maxclust', t=Maxclust)
unique, counts = np.unique(clsH, return_counts=True)
RandGMM[1], JaccardGMM[1], NMIGMM[1] = clusterval(y,clsH)
RandGMM[2], JaccardGMM[2], NMIGMM[2] = clusterval(cls,clsH)

# Display dendrogram
max_display_levels=10
plt.figure(8,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
plt.show()

#%%
#######################################################
### PROBLEM B: OUTLIER DETECTION    ###################
#######################################################


#%%
#######################################################
### PROBLEM C: ASSOCIATION MINING   ###################
#######################################################

numeric_attribute_names      = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']
categorical_attributes_names = ['famhist_present','chd'] 

## Set control parameters
numeric_var_splits = 4 # This var determines the number of splits for binarization of the numeric variable. 2 corresponds to a 50:50 split, 4 to 25:25:25:25 split, etc. 

## Define helping functions
def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

def apriori_rules_as_df(rules,y_itemsets):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:
            if(len(set(y_itemsets).intersection(list( o.items_add)))>0):
                conf = round(o.confidence,ndigits=2)
                supp = round(r.support,ndigits=2)
                x = ", ".join( list( o.items_base ) )
                y = ", ".join( list( o.items_add ) )
                #print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
                frules.append( (x,y,supp,conf) )
    df_frules = pd.DataFrame(frules,columns=['X','Y','support','confidence'])\
    .sort_values(by='support', ascending = False)\
    .assign(lift = lambda df:round(df.confidence / df.support,ndigits=1))\
    .reset_index().drop(columns='index')            
    return df_frules

# create quantile arrays
quantiles       = np.linspace(0,1,numeric_var_splits+1)
quantiles_lower = quantiles[0:numeric_var_splits]
quantiles_upper = quantiles[1:]

# Run loop to create binarized data frame
df_binarized = pd.DataFrame()
for var in numeric_attribute_names:
    #var = 'sbp'
    df_tmp = df_heart_disease[[var]].assign(tmp_var = "")
    for quantile_lower,quantile_upper in zip(quantiles_lower,quantiles_upper):
        low_quantile_value  = df_tmp[[var]].quantile(quantile_lower).values[0]
        high_quantile_value = df_tmp[[var]].quantile(quantile_upper).values[0]
        df_tmp.loc[(df_tmp[var]>=low_quantile_value) & (df_tmp[var]<high_quantile_value),'tmp_var'] = str(int(quantile_lower*100)) + "%_to_" + str(int(quantile_upper*100)) + "%"
 
    df_tmp = pd.get_dummies(df_tmp['tmp_var'], prefix=var,drop_first=True)
    df_binarized = pd.concat([df_binarized,df_tmp],axis = 1)

for var in categorical_attributes_names:
    df_binarized = pd.concat([df_binarized,df_heart_disease[[var]]], axis = 1)
    df_binarized = df_binarized.assign(tmp_var = lambda df:1 - df[var])
    df_binarized = df_binarized.rename(columns={var:var+"_yes",'tmp_var':var+"_no"})
    
X = df_binarized.to_numpy(dtype=np.int32)


# Transform binarized data to label values
T = mat2transactions(X, labels = list(df_binarized.columns))

# find association rules with apriory algorithm
rules = apriori(T, min_support=0.05, min_confidence=.6)
y_itemsets   = ['chd_yes']
df_chd_rules = apriori_rules_as_df(rules,y_itemsets)

## Save df table as latex format
with pd.option_context("max_colwidth",1000):
    latex_table  = df_chd_rules.to_latex(index=False, column_format='p{7.5cm}rrrr')
with open('association_rules_table.tex','w') as tf:
    tf.write(latex_table)