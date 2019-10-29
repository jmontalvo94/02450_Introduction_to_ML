# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:05:50 2019

@author: Emil Chrisander, Julian Böhm, and Jorge M. Arvizu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
import pdb
from scipy import stats ## We use the mode function in the stats module


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

## Define attribute names
attribute_names = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age','famhist_present']
class_name      = ['chd']

#######################################################
### STANDARDIZATION OF ATTRIBUTES #####################
#######################################################

## Start by creating a matric representation of the dataframe (only keep attributes)
X = df_heart_disease[attribute_names].to_numpy(dtype=np.float32) ## Type is set to float to allow for math calculations
y = df_heart_disease[class_name].to_numpy(dtype=np.float32)

## Store dimensions of X as local variables
N = np.shape(X)[0] ## Number of observations
M = np.shape(X)[1] ## Number of attributes

## Substract mean values from X (create X_tilde)
X_tilde = X - np.ones((N,1))*X.mean(axis=0)

## Divide by std. deviation from X_tilde
X_tilde = X_tilde*(1/np.std(X_tilde,0))

## Overwrite X_tilde as X, since we won't be using the non-standardized X
X = X_tilde


#######################################################
### PROBLEM 1a: REGRESSION ANALYSIS ###################
#######################################################


#######################################################
### PROBLEM 1b: REGRESSION ANALYSIS ###################
#######################################################


#######################################################
### PROBLEM 2: CLASSIFICATION #########################
#######################################################

## Set general options
print_cv_inner_loop_text = False
print_cv_outer_loop_text = True

## Select the applied classifiers
apply_KNN = True ## if set to True a KNN model is applied to the data. Remember to set regularization options

## Regularization options
min_k_KNN = 1 ## The minimum number of neighbours
max_k_KNN = 20 ## The maximum number of neighbours

## KNN options (only needed if apply_KNN is set to True)
dist=2 # Distance metric (corresponds to 2nd norm, euclidean distance). You can set dist=1 to obtain manhattan distance (cityblock distance).
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski


## DTC options (only needed if apply_DTC is set to True)
DTC_criterion = 'gini'

## Set K-folded CV options 
K_1   = 10 # Number of outer loops
K_2   = 10 # Number of inner loops
CV_1  = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True)
CV_2  = sklearn.model_selection.KFold(n_splits=K_2,shuffle=True)


## Define holders for outer CV results
test_error_outer_baseline         = [] ## Store validation error (the inner) of the baseline model
test_error_outer_KNN              = [] ## Store validation error (the inner) of the baseline model
data_outer_test_length            = []
optimal_regularization_param_KNN  = []

## Outer loop
k_outer = 0
for train_outer_index, test_outer_index in CV_1.split(X):
    if(print_cv_outer_loop_text):
        print('Computing CV outer fold: {0}/{1}..'.format(k_outer+1,K_1))
    X_train_outer, y_train_outer = X[train_outer_index,:], y[train_outer_index]
    X_test_outer, y_test_outer = X[test_outer_index,:], y[test_outer_index]
    
    ## Save length of outer train and test data
    data_outer_train_length    = float(len(y_train_outer))
    data_outer_test_length_tmp = float(len(y_test_outer))
    
    ## Define holders for inner CV results
    best_inner_model_baseline = []
    best_inner_model_KNN      = []
    error_inner_baseline      = [] ## Store validation error (the inner)
    error_inner_KNN           = []
    data_validation_length    = [] ## Store the length of D^val 
    
    ## Validation errors matrices (only for non-baseline models as baseline model does not test different models)
    validation_errors_inner_KNN_matrix  = np.array(np.ones(max_k_KNN)) ## This 1d array is used to vertical stack with validation error 1d arrays for each s KNN model. It is erased once these have been stacked into one matrix.
    
    ## Inner loop
    k_inner=0
    for train_inner_index, test_inner_index in CV.split(X_train_outer):
        if(print_cv_inner_loop_text):
            print('Computing CV inner fold: {0}/{1}..'.format(k_inner+1,K_2))
    
        ## Extract training and test set for current CV fold
        X_train_inner, y_train_inner = X[train_inner_index,:], y[train_inner_index]
        X_test_inner, y_test_inner = X[test_inner_index,:], y[test_inner_index]        
                  
        ## 'Fit' baseline model (chooses the class with most obs - aka. the mode in statistics)
        best_inner_model_baseline_tmp = stats.mode(y_train_inner).mode[0][0]
        y_est_test_inner_baseline     = np.ones((y_test_inner.shape[0],1))*best_inner_model_baseline_tmp
        
        ## Calculate validation error over inner test data
        validation_errors_inner_baseline = np.sum(y_est_test_inner_baseline != y_test_inner) / float(len(y_test_inner))
                
        ## Store best fitted model
        best_inner_model_baseline.append(best_inner_model_baseline_tmp)
        
        ## Store accuracy of CV-loop
        error_inner_baseline.append(validation_errors_inner_baseline)
        
        ## Store data validation length
        data_validation_length.append(float(len(y_test_inner)))
        
        ## Estimate KNN if apply_KNN is true
        if (apply_KNN):
            validation_errors_inner_KNN   = []
            for k_nearest_neighbour_tmp in range(min_k_KNN,max_k_KNN + 1):
                # Fit classifier and classify the test points
                knclassifier = KNeighborsClassifier(n_neighbors=k_nearest_neighbour_tmp, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
                
                knclassifier.fit(X_train_inner.squeeze(), y_train_inner.squeeze()) ## knclassifier.fit requires .squeeze of input matrices
                
                y_est_inner_model_KNN_tmp       = knclassifier.predict(X_test_inner)
                validation_errors_inner_KNN_tmp = np.sum(y_est_inner_model_KNN_tmp != y_test_inner.squeeze()) / float(len(y_test_inner))
                validation_errors_inner_KNN.append(validation_errors_inner_KNN_tmp)
            
            validation_errors_inner_KNN = np.array(validation_errors_inner_KNN)
            validation_errors_inner_KNN_matrix = np.vstack((validation_errors_inner_KNN_matrix,validation_errors_inner_KNN))

                
        ## add 1 to inner counter
        k_inner+=1
        
    ## Estimate generalization error of each model    
    generalized_error_inner_baseline_model = np.sum(np.multiply(data_validation_length,error_inner_baseline)) * (1/data_outer_train_length)
          
    ## 'Fit' baseline model on outside data (chooses the class with most obs - aka. the mode in statistics)
    best_outer_model_baseline_tmp = stats.mode(y_train_outer).mode[0][0]
    y_est_test_outer_baseline     = np.ones((y_test_outer.shape[0],1))*best_outer_model_baseline_tmp
      
    ## Estimate the test error (best model from inner fitted on the outer data)
    test_error_outer_baseline_tmp = np.sum(y_est_test_outer_baseline != y_test_outer) / float(len(y_test_outer))
    test_error_outer_baseline.append(test_error_outer_baseline_tmp)
    
    ## Add length of outer test data
    data_outer_test_length.append(data_outer_test_length_tmp)
    
    ## Find optimal model of KNN (if apply_KNN is true)
    if(apply_KNN):        
        validation_errors_inner_KNN_matrix = np.delete(validation_errors_inner_KNN_matrix,0,0) ## Removes the first 1d array with ones.
        validation_errors_inner_KNN_matrix = np.transpose(validation_errors_inner_KNN_matrix) ## Need to transpose validation_errors_inner_KNN_matrix, such that the dimensions are (20 x 10). That is, a vector for each models performance on the inner loop CV) 
        estimated_inner_test_error_KNN_models = []
        for s in range(0,len(validation_errors_inner_KNN_matrix)):
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_KNN_matrix[s])) / data_outer_train_length
            estimated_inner_test_error_KNN_models.append(tmp_inner_test_error)
        
        ## Saves the regularization parameter for the best performing KNN model
        lowest_est_inner_error_KNN_models = min(estimated_inner_test_error_KNN_models)
        optimal_regularization_param_KNN.append(list(estimated_inner_test_error_KNN_models).index(lowest_est_inner_error_KNN_models) + 1) # Plus one since list position starts at 0.
        
        ## Estimates the test error on outer test data
        knclassifier = KNeighborsClassifier(n_neighbors=optimal_regularization_param_KNN[k_outer], p=dist, 
                            metric=metric,
                            metric_params=metric_params)
        
        knclassifier.fit(X_train_outer.squeeze(), y_train_outer.squeeze()) ## knclassifier.fit requires .squeeze of input matrices
        
        y_est_outer_model_KNN_tmp       = knclassifier.predict(X_test_outer)
        test_error_outer_KNN_tmp        = np.sum(y_est_outer_model_KNN_tmp != y_test_outer.squeeze()) / float(len(y_test_outer))
        test_error_outer_KNN.append(test_error_outer_KNN_tmp)

    
    ## Add 1 to outer counter
    k_outer+=1


## Estimate the generalization error
generalization_error_baseline_model = np.sum(np.multiply(test_error_outer_baseline,data_outer_test_length)) * (1/N)   
if (apply_KNN):
    generalization_error_KNN_model = np.sum(np.multiply(test_error_outer_KNN,data_outer_test_length)) * (1/N)   
