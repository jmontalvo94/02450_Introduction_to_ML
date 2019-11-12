# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:05:50 2019

@author: Emil Chrisander, Julian Böhm, and Jorge Montalvo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary,correlated_ttest, rlr_validate
import torch
#import pdb
from scipy import stats ## We use the mode function in the stats module

## Set random seed to ensure same results whenever CV is performed
random_seed = 1337

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
attribute_names = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age','famhist_present']
class_name      = ['chd']


#%%
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

#%%
#######################################################
### PROBLEM 1a: REGRESSION ANALYSIS ###################
#######################################################



#%%
#######################################################
### PROBLEM 1b: REGRESSION ANALYSIS ###################
#######################################################

## Define attribute names
attribute_names = ['sbp','tobacco','chd','adiposity','typea','obesity','alcohol','age','famhist_present']
class_name      = ['ldl']

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

#%%

## Set general options
print_cv_inner_loop_text = True
print_cv_outer_loop_text = True

## Select the applied classifiers
apply_baseline = False
apply_ANN      = True ## if set to True a ANN model is applied to the data. Remember to set regularization options
apply_linear   = True

## Select statistic test type
apply_setup_ii = True

## Regularization options ANN
min_n_hidden_units = 1 ## The minimum number of hidden units
max_n_hidden_units = 3 ## The maximum number of hidden units

## Regularization options Linear Regresssion
#lambda_interval = np.logspace(0, 3, 30)     ## Value from 0 to 1000 seems reasonble according to estimates in the previous sub task.
lambda_interval = np.array(range(0,210,10)) ## Value from 0 to 1000 seems reasonble according to estimates in the previous sub task.

## ANN options
loss_fn   = torch.nn.MSELoss()
max_iter  = 50000
n_rep_ann = 1

## Set K-folded CV options 
K_1   = 10 # Number of outer loops
K_2   = 10 # Number of inner loops
CV_1  = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed)
CV_2  = sklearn.model_selection.KFold(n_splits=K_2,shuffle=True, random_state = random_seed)
CV_setup_ii = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed + 1) ## Ensures that the CV for setup ii test is never the same randomization as for the estimation CVs


#%%
## Define holders for outer CV results
test_error_outer_baseline                = [] ## Store validation error (the inner) of the baseline model
test_error_outer_linear                  = [] ## Store validation error (the inner) of the baseline model
test_errors_outer_ANN                    = []
data_outer_test_length                   = []
optimal_regularization_param_baseline    = []
optimal_regularization_param_linear      = []
optimal_regularization_param_ANN         = []

## Outer loop
k_outer = 0
for train_outer_index, test_outer_index in CV_1.split(X):
    if(print_cv_outer_loop_text):
        print('Computing CV outer fold: {0}/{1}..'.format(k_outer+1,K_1))
    X_train_outer, y_train_outer = X[train_outer_index,:], y[train_outer_index]
    X_test_outer, y_test_outer = X[test_outer_index,:], y[test_outer_index]
    
    if (apply_ANN):
        X_train_outer_tensor = torch.tensor(X[train_outer_index,:], dtype=torch.float)
        y_train_outer_tensor = torch.tensor(y[train_outer_index], dtype=torch.float)
        X_test_outer_tensor  = torch.tensor(X[test_outer_index,:], dtype=torch.float)
        y_test_outer_tensor  = torch.tensor(y[test_outer_index], dtype=torch.uint8)
    
    ## Save length of outer train and test data
    data_outer_train_length    = float(len(y_train_outer))
    data_outer_test_length_tmp = float(len(y_test_outer))
    
    ## Define holders for inner CV results
    best_inner_model_baseline      = []
    error_inner_baseline           = [] ## Store validation error (the inner)
    data_validation_length         = [] ## Store the length of D^val 

    
    ## Validation errors matrices (only for non-baseline models as baseline model does not test different models)
    validation_errors_inner_ANN_matrix         = np.array(np.ones(max_n_hidden_units - min_n_hidden_units + 1)) ## This 1d array is used to vertical stack with validation error 1d arrays for each s KNN model. It is erased once these have been stacked into one matrix.
    validation_errors_inner_linear_matrix      = np.array(np.ones(len(lambda_interval))) ## This 1d array is used to vertical stack with validation error 1d arrays for each s logistic models. It is erased once these have been stacked into one matrix.
    hidden_units_matrix                        = np.array(np.ones(max_n_hidden_units - min_n_hidden_units + 1)) ## This is used to store the regularization parameter for the ANN 
    regularization_param_linear_matrix         = np.array(np.ones(len(lambda_interval)))
        
    ## Inner loop
    k_inner=0
    for train_inner_index, test_inner_index in CV_2.split(X_train_outer):
        if(print_cv_inner_loop_text):
            print('Computing CV inner fold: {0}/{1}..'.format(k_inner+1,K_2))
    
        ## Extract training and test set for current CV fold
        X_train_inner, y_train_inner = X[train_inner_index,:], y[train_inner_index]
        X_test_inner, y_test_inner = X[test_inner_index,:], y[test_inner_index]        
                  
        if (apply_ANN):
            X_train_inner_tensor = torch.tensor(X[train_inner_index,:], dtype=torch.float)
            y_train_inner_tensor = torch.tensor(y[train_inner_index], dtype=torch.float)
            X_test_inner_tensor = torch.tensor(X[test_inner_index,:], dtype=torch.float)
            y_test_inner_tensor = torch.tensor(y[test_inner_index], dtype=torch.uint8)
        
        
        ## 'Fit' baseline model (simply the unconditional mean value of y)
        mean_y                        = np.mean(y_train_inner)
        y_est_test_inner_baseline     = mean_y
        
        ## Calculate validation error over inner test data
        validation_errors_inner_baseline = np.sum((y_est_test_inner_baseline - y_test_inner)**2) / float(len(y_test_inner)) 
        
        ## Store best fitted model
        #best_inner_model_baseline.append(best_inner_model_baseline_tmp)
        
        ## Store accuracy of CV-loop
        error_inner_baseline.append(validation_errors_inner_baseline)
        
        ## Store data validation length
        data_validation_length.append(float(len(y_test_inner)))
        

         ## Estimate linear regression if apply_linear is true
        if (apply_linear):
         
             
            validation_errors_inner_linear  = []
            regularization_param_linear     = []
             
            for lambda_val in lambda_interval:
            ## Use linear model with regularization parameter (aka ridge model (see: https://scikit-learn.org/stable/modules/linear_model.html))
                model = sklearn.linear_model.Ridge(alpha=lambda_val)
                model = model.fit(X_train_inner,y_train_inner)
                y_est_test_inner_linear = model.predict(X_test_inner)
                 
                error      = (y_est_test_inner_linear - y_test_inner)**2
                error_rate =  np.sum(error) / len(y_test_inner)
                validation_errors_inner_linear.append(error_rate)
                regularization_param_linear.append(lambda_val)
                
            validation_errors_inner_linear        = np.array(validation_errors_inner_linear)
            validation_errors_inner_linear_matrix = np.vstack((validation_errors_inner_linear_matrix,validation_errors_inner_linear))
            regularization_param_linear_matrix    = np.vstack((regularization_param_linear_matrix,regularization_param_linear))     
                
        ## Estimate ANN if apply_ANN is true
        if (apply_ANN):
            validation_errors_inner_ANN  = []
            hidden_unit_applied          = []
            for n_hidden_units in range(min_n_hidden_units,max_n_hidden_units + 1):
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hidden units
                    # 1st transfer function, either Tanh or ReLU:
                    #torch.nn.ReLU(), 
                    torch.nn.Tanh(),   
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
                
                # Run optimization
                net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X_train_inner_tensor,
                                                   y=y_train_inner_tensor,
                                                   n_replicates=n_rep_ann,
                                                   max_iter=max_iter)
                
                # Determine estimated class labels for test set
                y_est_inner   = net(X_test_inner_tensor) # activation of final note, i.e. prediction of network
                
                # Determine errors and error rate
                e = (y_est_inner.float()-y_test_inner_tensor.float())**2
                error_rate = (sum(e).type(torch.float)/len(y_test_inner_tensor)).data.numpy()[0]
                validation_errors_inner_ANN.append(error_rate)
                
                ## Add applied hidden units to array
                hidden_unit_applied.append(n_hidden_units)
                
            validation_errors_inner_ANN        = np.array(validation_errors_inner_ANN)
            validation_errors_inner_ANN_matrix = np.vstack((validation_errors_inner_ANN_matrix,validation_errors_inner_ANN))
            hidden_units_matrix                = np.vstack((hidden_units_matrix,hidden_unit_applied))
            
            
            
        ## add 1 to inner counter
        k_inner+=1
        
    ## Estimate generalization error of each model    
    #generalized_error_inner_baseline_model = np.sum(np.multiply(data_validation_length,error_inner_baseline)) * (1/data_outer_train_length)
          
    ## 'Fit' baseline model on outside data (chooses the average value of the observations in the test set)
    #best_outer_model_baseline_tmp = sum(y_train_outer)/float(len(y_train_outer))
    #y_est_test_outer_baseline     = np.ones((y_test_outer.shape[0],1))*best_outer_model_baseline_tmp
      
    ## 'Fit' baseline model 
    mean_y                        = np.mean(y_train_outer)
    y_est_test_outer_baseline     = mean_y     
    
    ## Estimate the test error (best model from inner fitted on the outer data)
    test_error_outer_baseline_tmp = np.sum((y_est_test_outer_baseline - y_test_outer)**2) / float(len(y_test_outer))
    test_error_outer_baseline.append(test_error_outer_baseline_tmp)
    

    
    ## Calculate validation error over inner test data
    validation_errors_inner_baseline = np.sum((y_est_test_inner_baseline - y_test_inner)**2) / float(len(y_test_inner)) 
    
    
    
    
    ## Add length of outer test data
    data_outer_test_length.append(data_outer_test_length_tmp)
    
    ## Find optimal model of ANN (if apply_ANN is true)
    if (apply_ANN):        
        validation_errors_inner_ANN_matrix = np.delete(validation_errors_inner_ANN_matrix,0,0) ## Removes the first 1d array with ones.
        hidden_units_matrix                = np.delete(hidden_units_matrix,0,0)
        validation_errors_inner_ANN_matrix = np.transpose(validation_errors_inner_ANN_matrix) ## Need to transpose validation_errors_inner_KNN_matrix, such that the dimensions are (20 x 10). That is, a vector for each models performance on the inner loop CV) 
        estimated_inner_test_error_ANN_models = []
        for s in range(0,len(validation_errors_inner_ANN_matrix)):
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_ANN_matrix[s])) / data_outer_train_length
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_ANN_matrix[s])) / data_outer_train_length

            estimated_inner_test_error_ANN_models.append(tmp_inner_test_error)
        
        ## Saves the regularization parameter for the best performing ANN model
        lowest_est_inner_error_ANN_models = min(estimated_inner_test_error_ANN_models)
        index_tmp                         = (list(estimated_inner_test_error_ANN_models).index(lowest_est_inner_error_ANN_models)) # Plus one since list position starts at 0.       
        optimal_regularization_param_ANN.append(hidden_units_matrix[k_outer][index_tmp])
        
        ## Estimates the test error on outer test data
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, optimal_regularization_param_ANN[k_outer].astype(int)), #M features to H hidden units
            # 1st transfer function, either Tanh or ReLU:
            #torch.nn.ReLU(), 
            torch.nn.Tanh(),   
            torch.nn.Linear(optimal_regularization_param_ANN[k_outer].astype(int), 1), # H hidden units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
            )
        
        ## Run optimization
        net, final_loss, learning_curve = train_neural_net(model,
                                           loss_fn,
                                           X=X_train_outer_tensor,
                                           y=y_train_outer_tensor,
                                           n_replicates=n_rep_ann,
                                           max_iter=max_iter)
        
        
        
        # Determine estimated class labels for test set
        y_est_outer_ANN = net(X_test_outer_tensor) # activation of final note, i.e. prediction of network
        
        # Determine errors and error rate
        e = (y_est_outer_ANN.float()-y_test_outer_tensor.float())**2
        error_rate = (sum(e).type(torch.float)/len(y_test_outer_tensor)).data.numpy()[0]
        test_errors_outer_ANN.append(error_rate)


        if (apply_linear):
          validation_errors_inner_linear_matrix = np.delete(validation_errors_inner_linear_matrix,0,0) ## Removes the first 1d array with ones.
          # Investigate relationsship between lampda and validation errors (tmp's first element is the lampda value. The second element is the validation error)
          #tmp = np.vstack((lambda_interval,np.mean(validation_errors_inner_logistics_matrix,axis=0))).T
   

          ## calculates the test error for each model s of the linear models (accross s lambda reg. parameters) 
          validation_errors_inner_linear_matrix   = np.transpose(validation_errors_inner_linear_matrix)
          estimated_inner_test_error_linear_models = []
          for s in range(0,len(validation_errors_inner_linear_matrix)):
              tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_linear_matrix[s])) / data_outer_train_length
              estimated_inner_test_error_linear_models.append(tmp_inner_test_error)
          
          
          ## Saves the regularization parameter for the best performing linear model
          lowest_est_inner_error_linear_models = min(estimated_inner_test_error_linear_models)
          index_lambda = list(estimated_inner_test_error_linear_models).index(lowest_est_inner_error_linear_models) # Plus one since list position starts at 0.
          optimal_regularization_param_linear.append(lambda_interval[index_lambda])         
         
          ## Estimate the test error on the outer test data
          model = sklearn.linear_model.Ridge(alpha=optimal_regularization_param_linear[k_outer])
          model = model.fit(X_train_outer,y_train_outer)
          y_est_test_outer_linear = model.predict(X_test_outer)
         
          error      = (y_est_test_outer_linear - y_test_outer)**2
          error_rate =  np.sum(error) / len(y_test_outer)
          test_error_outer_linear.append(error_rate)  
         
             
    ## Add 1 to outer counter
    k_outer+=1
    
## Estimate the generalization error
generalization_error_baseline_model = np.sum(np.multiply(test_error_outer_baseline,data_outer_test_length)) * (1/N) 
print('est gen error of baseline model: ' +str(round(generalization_error_baseline_model, ndigits=3)))  
if (apply_ANN):
    generalization_error_ANN_model = np.sum(np.multiply(test_errors_outer_ANN,data_outer_test_length)) * (1/N)
    print('est gen error of ANN model: ' +str(round(generalization_error_ANN_model, ndigits=3)))    

if (apply_linear):
    generalization_error_linear_model = np.sum(np.multiply(test_error_outer_linear,data_outer_test_length)) * (1/N)
    print('est gen error of linear model: ' +str(round(generalization_error_linear_model, ndigits=3)))


#%%
    
## Create output table as dataframe
n_of_cols                  = sum([apply_ANN,apply_linear])*2 + 2 ## the + 2 is the baseline model which is always included and test data size   
n_of_index                 = K_1 + 1 ## Plus one is for the final row which is the generalized error estimate
df_output_table            = pd.DataFrame(np.ones((n_of_index,n_of_cols)),index=range(1,n_of_index + 1))
df_output_table.index.name = "Outer fold"
   
    
if(apply_ANN):
    df_output_table.columns                = ['test_data_size','n_hidden_units','ANN_test_error','lambda','Linear_test_error','baseline_test_error']
    optimal_regularization_param_ANN.append('')
    optimal_regularization_param_linear.append('')
    data_outer_test_length.append('')
    col_2                                  = list(np.array(test_errors_outer_ANN).round(3)*100)
    col_2.append(round(generalization_error_ANN_model*100,ndigits=1))
    col_4                                  = list(np.array(test_error_outer_linear).round(3)*100)
    col_4.append(round(generalization_error_linear_model*100,ndigits=1))    
    col_5                                  = list(np.array(test_error_outer_baseline).round(3)*100)
    col_5.append(round(generalization_error_baseline_model*100,ndigits=1))       
        
    ## Add values to columns in output table    
    df_output_table['test_data_size']      = data_outer_test_length
    df_output_table['n_hidden_units']      = optimal_regularization_param_ANN
    df_output_table['ANN_test_error']      = col_2
    df_output_table['lambda']              = optimal_regularization_param_linear
    df_output_table['Linear_test_error']   = col_4
    df_output_table['baseline_test_error'] = col_5
    

## Export as csv
df_output_table.to_csv('Regression_summary_table_50000_10.csv')

#%%

### Statistical Test Evaluation (SETUP II)

## Statistical test settings
loss_in_r_function = 2 ## This implies the loss is squared in the r_j formula of box 11.4.1 
r_baseline_vs_linear  = []                 ## The list to keep the r test size 
r_baseline_vs_ANN = []                 ## The list to keep the r test size 
r_ANN_vs_linear = []                 ## The list to keep the r test size 
alpha_t_test            = 0.05
rho_t_test              = 1/K_1

if(apply_setup_ii):
    most_common_lambda    = stats.mode(optimal_regularization_param_linear).mode[0].astype('float64')    
    y_true = []
    yhat = []
    
    k = 0
    for train_index,test_index in CV_setup_ii.split(X):
        print('Computing setup II CV K-fold: {0}/{1}..'.format(k+1,K_1))
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        
        X_train_tensor = torch.tensor(X[train_index,:], dtype=torch.float)
        y_train_tensor = torch.tensor(y[train_index], dtype=torch.float)
        X_test_tensor = torch.tensor(X[test_index,:], dtype=torch.float)
        y_test_tensor = torch.tensor(y[test_index], dtype=torch.uint8)
        
        model_baseline = np.mean(y_train)      
        model_linear = sklearn.linear_model.Ridge(alpha=most_common_lambda).fit(X_train,y_train.squeeze())
        
        yhat_baseline  = np.ones((y_test.shape[0],1))*model_baseline.squeeze()
        yhat_linear  = model_linear.predict(X_test).reshape(-1,1) ## use reshape to ensure it is a nested array
            
        if(apply_ANN):
            most_common_regu_ANN  = 1
            model_second = lambda: torch.nn.Sequential(
                                    torch.nn.Linear(M, most_common_regu_ANN), #M features to H hidden units
                                    # 1st transfer function, either Tanh or ReLU:
                                    #torch.nn.ReLU(), 
                                    torch.nn.Tanh(),   
                                    torch.nn.Linear(most_common_regu_ANN, 1), # H hidden units to 1 output neuron
                                    # no final tranfer function, i.e. "linear output"
                                    )
        
            ## Run optimization
            net, final_loss, learning_curve = train_neural_net(model_second,
                                               loss_fn,
                                               X=X_train_tensor,
                                               y=y_train_tensor,
                                               n_replicates=n_rep_ann,
                                               max_iter=max_iter)
            
            # Determine estimated regression value for test set
            yhat_ANN = net(X_test_tensor)
            yhat_ANN = yhat_ANN.detach().numpy()
        
        ## Add true classes and store estimated classes    
        y_true.append(y_test)
        yhat.append(np.concatenate([yhat_baseline, yhat_linear,yhat_ANN], axis=1) )
        
        ## Compute the r test size and store it
        r_baseline_vs_linear.append( np.mean( np.abs( yhat_baseline-y_test ) ** loss_in_r_function - np.abs( yhat_linear-y_test) ** loss_in_r_function ) )
        r_baseline_vs_ANN.append( np.mean( np.abs( yhat_baseline-y_test ) ** loss_in_r_function - np.abs( yhat_ANN-y_test) ** loss_in_r_function ) )
        r_ANN_vs_linear.append( np.mean( np.abs( yhat_ANN-y_test ) ** loss_in_r_function - np.abs( yhat_linear-y_test) ** loss_in_r_function ) )
        
        ## add to counter
        k += 1


    ## Baseline vs logistic regression    
    p_setupII_base_vs_linear, CI_setupII_base_vs_linear = correlated_ttest(r_baseline_vs_linear, rho_t_test, alpha=alpha_t_test)
    
    ## Baseline vs 2nd model    
    p_setupII_base_vs_ANN, CI_setupII_base_vs_ANN = correlated_ttest(r_baseline_vs_ANN, rho_t_test, alpha=alpha_t_test)
    
    ## Logistic regression vs 2nd model    
    p_setupII_ANN_vs_linear, CI_setupII_ANN_vs_linear = correlated_ttest(r_ANN_vs_linear, rho_t_test, alpha=alpha_t_test)

    ## Create output table for statistic tests
    df_output_table_statistics = pd.DataFrame(np.ones((3,5)), columns = ['H_0','p_value','CI_lower','CI_upper','conclusion'])
    df_output_table_statistics[['H_0']] = ['err_baseline-err_linear=0','err_ANN-err_linear=0','err_baseline-err_ANN=0']
    df_output_table_statistics[['p_value']]         = [p_setupII_base_vs_linear,p_setupII_ANN_vs_linear,p_setupII_base_vs_ANN]
    df_output_table_statistics[['CI_lower']]        = [CI_setupII_base_vs_linear[0],CI_setupII_ANN_vs_linear[0],CI_setupII_base_vs_ANN[0]]
    df_output_table_statistics[['CI_upper']]        = [CI_setupII_base_vs_linear[1],CI_setupII_ANN_vs_linear[1],CI_setupII_base_vs_ANN[1]]
    rejected_null                                   = (df_output_table_statistics.loc[:,'p_value']<alpha_t_test)
    df_output_table_statistics.loc[rejected_null,'conclusion']   = 'H_0 rejected'
    df_output_table_statistics.loc[~rejected_null,'conclusion']  = 'H_0 not rejected'
    df_output_table_statistics                      = df_output_table_statistics.set_index('H_0')
    
    ## Export df as csv
    df_output_table_statistics.to_csv('Regression_statistic_test_50000_10.csv',encoding='UTF-8')

#%%
#######################################################
### PROBLEM 2: CLASSIFICATION #########################
#######################################################

## Define attribute names
attribute_names = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age','famhist_present']
class_name      = ['chd']

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

## Set general options
print_cv_inner_loop_text = False
print_cv_outer_loop_text = True

## Select the applied classifiers
apply_KNN      = True ## if set to True a KNN model is applied to the data. Remember to set regularization options
apply_logistic = True 
apply_ANN      = False

## Select statistic test type
apply_setup_ii = True

## Regularization options KNN
min_k_KNN = 1 ## The minimum number of neighbours
max_k_KNN = 50 ## The maximum number of neighbours

## Regularization options ANN
min_n_hidden_units = 1
max_n_hidden_units = 10

## Regularization options Logistic Regresssion
lambda_interval = np.logspace(-1, 3, 30)

## KNN options (only needed if apply_KNN is set to True)
dist          = 2 # Distance metric (corresponds to 2nd norm, euclidean distance). You can set dist=1 to obtain manhattan distance (cityblock distance).
metric        = 'minkowski'
metric_params = {} # no parameters needed for minkowski

## ANN options
loss_fn   = torch.nn.BCELoss()
max_iter  = 10000
n_rep_ann = 2

## Set K-folded CV options 
K_1   = 10 # Number of outer loops
K_2   = 10 # Number of inner loops
CV_1  = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed)
CV_2  = sklearn.model_selection.KFold(n_splits=K_2,shuffle=True, random_state = random_seed)
CV_setup_ii = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed + 1) ## Ensures that the CV for setup ii test is never the same randomization as for the estimation CVs

## Statistical test settings
loss_in_r_function = 2 ## This implies the loss is squared in the r_j formula of box 11.4.1 
r_baseline_vs_logistic  = []                 ## The list to keep the r test size 
r_baseline_vs_sec_model = []                 ## The list to keep the r test size 
r_sec_model_vs_logistic = []                 ## The list to keep the r test size 
alpha_t_test            = 0.05
rho_t_test              = 1/K_1


## Define holders for outer CV results
test_error_outer_baseline                = [] ## Store validation error (the inner) of the baseline model
test_error_outer_KNN                     = [] ## Store validation error (the inner) of the baseline model
test_errors_outer_ANN                    = []
test_errors_outer_logistics              = []
data_outer_test_length                   = []
optimal_regularization_param_KNN         = []
optimal_regularization_param_ANN         = []
optimal_regularization_param_logitisctic = []

## Outer loop
k_outer = 0
for train_outer_index, test_outer_index in CV_1.split(X):
    if(print_cv_outer_loop_text):
        print('Computing CV outer fold: {0}/{1}..'.format(k_outer+1,K_1))
    X_train_outer, y_train_outer = X[train_outer_index,:], y[train_outer_index]
    X_test_outer, y_test_outer = X[test_outer_index,:], y[test_outer_index]
    
    if (apply_ANN):
        X_train_outer_tensor = torch.tensor(X[train_outer_index,:], dtype=torch.float)
        y_train_outer_tensor = torch.tensor(y[train_outer_index], dtype=torch.float)
        X_test_outer_tensor  = torch.tensor(X[test_outer_index,:], dtype=torch.float)
        y_test_outer_tensor  = torch.tensor(y[test_outer_index], dtype=torch.uint8)
    
    ## Save length of outer train and test data
    data_outer_train_length    = float(len(y_train_outer))
    data_outer_test_length_tmp = float(len(y_test_outer))
    
    ## Define holders for inner CV results
    best_inner_model_baseline = []
    #best_inner_model_KNN      = []
    error_inner_baseline      = [] ## Store validation error (the inner)
    #error_inner_KNN           = []
    data_validation_length    = [] ## Store the length of D^val 
    
    ## Validation errors matrices (only for non-baseline models as baseline model does not test different models)
    validation_errors_inner_KNN_matrix         = np.array(np.ones(max_k_KNN-min_k_KNN + 1)) ## This 1d array is used to vertical stack with validation error 1d arrays for each s KNN model. It is erased once these have been stacked into one matrix.
    validation_errors_inner_ANN_matrix         = np.array(np.ones(max_n_hidden_units - min_n_hidden_units + 1)) ## This 1d array is used to vertical stack with validation error 1d arrays for each s KNN model. It is erased once these have been stacked into one matrix.
    validation_errors_inner_logistics_matrix   = np.array(np.ones(len(lambda_interval))) ## This 1d array is used to vertical stack with validation error 1d arrays for each s logistic models. It is erased once these have been stacked into one matrix.
    
    
    ## Inner loop
    k_inner=0
    for train_inner_index, test_inner_index in CV_2.split(X_train_outer):
        if(print_cv_inner_loop_text):
            print('Computing CV inner fold: {0}/{1}..'.format(k_inner+1,K_2))
    
        ## Extract training and test set for current CV fold
        X_train_inner, y_train_inner = X[train_inner_index,:], y[train_inner_index]
        X_test_inner, y_test_inner = X[test_inner_index,:], y[test_inner_index]        
                  
        if (apply_ANN):
            X_train_inner_tensor = torch.tensor(X[train_inner_index,:], dtype=torch.float)
            y_train_inner_tensor = torch.tensor(y[train_inner_index], dtype=torch.float)
            X_test_inner_tensor = torch.tensor(X[test_inner_index,:], dtype=torch.float)
            y_test_inner_tensor = torch.tensor(y[test_inner_index], dtype=torch.uint8)
        
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
                model = KNeighborsClassifier(n_neighbors=k_nearest_neighbour_tmp, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
                
                model = model.fit(X_train_inner.squeeze(), y_train_inner.squeeze()) ## knclassifier.fit requires .squeeze of input matrices
                
                y_est_inner_model_KNN_tmp       = model.predict(X_test_inner)
                validation_errors_inner_KNN_tmp = np.sum(y_est_inner_model_KNN_tmp != y_test_inner.squeeze()) / float(len(y_test_inner))
                validation_errors_inner_KNN.append(validation_errors_inner_KNN_tmp)
            
            validation_errors_inner_KNN = np.array(validation_errors_inner_KNN)
            validation_errors_inner_KNN_matrix = np.vstack((validation_errors_inner_KNN_matrix,validation_errors_inner_KNN))

        ## Estimate logistic regression if apply_logistic is true
        if (apply_logistic):
            validation_errors_inner_logistics  = []
            for s in range(0, len(lambda_interval)):
                model       = LogisticRegression(penalty='l2', C=1/lambda_interval[s], solver = 'liblinear')
                model       = model.fit(X_train_inner, y_train_inner.squeeze())            
                y_est_inner = model.predict(X_test_inner)
                
                validation_errors_inner_logistics_tmp = np.sum(y_est_inner != y_test_inner.squeeze()) / float(len(y_test_inner))
                validation_errors_inner_logistics.append(validation_errors_inner_logistics_tmp)
            
            validation_errors_inner_logistics = np.array(validation_errors_inner_logistics)
            validation_errors_inner_logistics_matrix = np.vstack((validation_errors_inner_logistics_matrix,validation_errors_inner_logistics))

                
        ## Estimate logistic regression if apply_logistic is true
        if (apply_ANN):
            validation_errors_inner_ANN  = []

            for n_hidden_units in range(min_n_hidden_units,max_n_hidden_units + 1):
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hidden units
                    # 1st transfer function, either Tanh or ReLU:
                    #torch.nn.ReLU(), 
                    torch.nn.Tanh(),   
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
                
                # Run optimization
                net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X_train_inner_tensor,
                                                   y=y_train_inner_tensor,
                                                   n_replicates=n_rep_ann,
                                                   max_iter=max_iter)
                
                # Determine estimated class labels for test set
                y_sigmoid   = net(X_test_inner_tensor) # activation of final note, i.e. prediction of network
                y_est_inner = y_sigmoid > .5 # threshold output of sigmoidal function
                
                # Determine errors and error rate
                e = (y_est_inner != y_test_inner_tensor)
                error_rate = (sum(e).type(torch.float)/len(y_test_inner_tensor)).data.numpy()[0]
                validation_errors_inner_ANN.append(error_rate)
                
                
                
            validation_errors_inner_ANN        = np.array(validation_errors_inner_ANN)
            validation_errors_inner_ANN_matrix = np.vstack((validation_errors_inner_ANN_matrix,validation_errors_inner_ANN))   
            
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
        
        y_est_outer_model_KNN       = knclassifier.predict(X_test_outer)
        test_error_outer_KNN_tmp        = np.sum(y_est_outer_model_KNN != y_test_outer.squeeze()) / float(len(y_test_outer))
        test_error_outer_KNN.append(test_error_outer_KNN_tmp)
        
        
    ## Find optimal model of ANN (if apply_ANN is true)
    if(apply_ANN):        
        validation_errors_inner_ANN_matrix = np.delete(validation_errors_inner_ANN_matrix,0,0) ## Removes the first 1d array with ones.
        validation_errors_inner_ANN_matrix = np.transpose(validation_errors_inner_ANN_matrix) ## Need to transpose validation_errors_inner_KNN_matrix, such that the dimensions are (20 x 10). That is, a vector for each models performance on the inner loop CV) 
        estimated_inner_test_error_ANN_models = []
        for s in range(0,len(validation_errors_inner_ANN_matrix)):
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_ANN_matrix[s])) / data_outer_train_length
            estimated_inner_test_error_ANN_models.append(tmp_inner_test_error)
        
        ## Saves the regularization parameter for the best performing KNN model
        lowest_est_inner_error_ANN_models = min(estimated_inner_test_error_ANN_models)
        optimal_regularization_param_ANN.append(list(estimated_inner_test_error_ANN_models).index(lowest_est_inner_error_ANN_models) + 1) # Plus one since list position starts at 0.
            
        
        ## Estimates the test error on outer test data
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, optimal_regularization_param_ANN[k_outer]), #M features to H hidden units
            # 1st transfer function, either Tanh or ReLU:
            #torch.nn.ReLU(), 
            torch.nn.Tanh(),   
            torch.nn.Linear(optimal_regularization_param_ANN[k_outer], 1), # H hidden units to 1 output neuron
            torch.nn.Sigmoid() # final tranfer function
            )
        
        ## Run optimization
        net, final_loss, learning_curve = train_neural_net(model,
                                           loss_fn,
                                           X=X_train_outer_tensor,
                                           y=y_train_outer_tensor,
                                           n_replicates=n_rep_ann,
                                           max_iter=max_iter)
        
        
        
        # Determine estimated class labels for test set
        y_sigmoid       = net(X_test_outer_tensor) # activation of final note, i.e. prediction of network
        y_est_outer_ANN = y_sigmoid > .5 # threshold output of sigmoidal function
        
        # Determine errors and error rate
        e = (y_est_outer_ANN != y_test_outer_tensor)
        error_rate = (sum(e).type(torch.float)/len(y_test_outer_tensor)).data.numpy()[0]
        test_errors_outer_ANN.append(error_rate)

    if(apply_logistic):
         validation_errors_inner_logistics_matrix = np.delete(validation_errors_inner_logistics_matrix,0,0) ## Removes the first 1d array with ones.
         # Investigate relationsship between lampda and validation errors (tmp's first element is the lampda value. The second element is the validation error)
         #tmp = np.vstack((lambda_interval,np.mean(validation_errors_inner_logistics_matrix,axis=0))).T
  

         ## calculates the test error for each model s of the logistic models (accross s lambda reg. parameters) 
         validation_errors_inner_logistics_matrix   = np.transpose(validation_errors_inner_logistics_matrix)
         estimated_inner_test_error_logistic_models = []
         for s in range(0,len(validation_errors_inner_logistics_matrix)):
             tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_logistics_matrix[s])) / data_outer_train_length
             estimated_inner_test_error_logistic_models.append(tmp_inner_test_error)
         
         
         ## Saves the regularization parameter for the best performing logit model
         lowest_est_inner_error_logistic_models = min(estimated_inner_test_error_logistic_models)
         index_lambda = list(estimated_inner_test_error_logistic_models).index(lowest_est_inner_error_logistic_models) # Plus one since list position starts at 0.
         optimal_regularization_param_logitisctic.append(lambda_interval[index_lambda])         
        
         ## Estimate the test error on the outer test data
         model                = LogisticRegression(penalty='l2', C=1/lambda_interval[index_lambda], solver = 'lbfgs')
         model                = model.fit(X_train_outer, y_train_outer.squeeze())            
         y_est_outer_logistic = model.predict(X_test_outer)
         
         test_errors_outer_logistics_tmp = np.sum(y_est_outer_logistic != y_test_outer.squeeze()) / float(len(y_test_outer))
         test_errors_outer_logistics.append(test_errors_outer_logistics_tmp)
         
    
    ## Add 1 to outer counter
    k_outer+=1


## Estimate the generalization error
generalization_error_baseline_model = np.sum(np.multiply(test_error_outer_baseline,data_outer_test_length)) * (1/N) 
print('est gen error of baseline model: ' +str(round(generalization_error_baseline_model, ndigits=3)))  
if (apply_KNN):
    generalization_error_KNN_model = np.sum(np.multiply(test_error_outer_KNN,data_outer_test_length)) * (1/N)
    print('est gen error of KNN model: ' +str(round(generalization_error_KNN_model, ndigits=3)))
if (apply_ANN):
    generalization_error_ANN_model = np.sum(np.multiply(test_errors_outer_ANN,data_outer_test_length)) * (1/N)
    print('est gen error of ANN model: ' +str(round(generalization_error_ANN_model, ndigits=3)))    
if (apply_logistic):
    generalization_error_logistic_model = np.sum(np.multiply(test_errors_outer_logistics,data_outer_test_length)) * (1/N)
    print('est gen error of logistic model: ' +str(round(generalization_error_logistic_model, ndigits=3)))
    
## Create output table as dataframe
n_of_cols                  = sum([apply_KNN,apply_ANN,apply_logistic])*2 + 2 ## the + 2 is the baseline model which is always included and test data size   
n_of_index                 = K_1 + 1 ## Plus one is for the final row which is the generalized error estimate
df_output_table            = pd.DataFrame(np.ones((n_of_index,n_of_cols)),index=range(1,n_of_index + 1))
df_output_table.index.name = "Outer fold"
if(apply_KNN):
    df_output_table.columns                = ['test_data_size','K','KNN_test_error','lambda','Logistic_test_error','baseline_test_error']
    optimal_regularization_param_KNN.append('')
    optimal_regularization_param_logitisctic.append('')
    data_outer_test_length.append('')
    col_2                                  = list(np.array(test_error_outer_KNN).round(3)*100)
    col_2.append(round(generalization_error_KNN_model*100,ndigits=1))
    col_4                                  = list(np.array(test_errors_outer_logistics).round(3)*100)
    col_4.append(round(generalization_error_logistic_model*100,ndigits=1))    
    col_5                                  = list(np.array(test_error_outer_baseline).round(3)*100)
    col_5.append(round(generalization_error_baseline_model*100,ndigits=1))       
        
    ## Add values to columns in output table
    df_output_table['test_data_size']      = data_outer_test_length    
    df_output_table['K']                   = optimal_regularization_param_KNN
    df_output_table['KNN_test_error']      = col_2
    df_output_table['lambda']              = optimal_regularization_param_logitisctic
    df_output_table['Logistic_test_error'] = col_4
    df_output_table['baseline_test_error'] = col_5

    
    
if(apply_ANN):
    df_output_table.columns                = ['test_data_size','n_hidden_units','ANN_test_error','lambda','Logistic_test_error','baseline_test_error']
    optimal_regularization_param_ANN.append('')
    optimal_regularization_param_logitisctic.append('')
    data_outer_test_length.append('')
    col_2                                  = list(np.array(test_errors_outer_ANN).round(3)*100)
    col_2.append(round(generalization_error_ANN_model*100,ndigits=1))
    col_4                                  = list(np.array(test_errors_outer_logistics).round(3)*100)
    col_4.append(round(generalization_error_logistic_model*100,ndigits=1))    
    col_5                                  = list(np.array(test_error_outer_baseline).round(3)*100)
    col_5.append(round(generalization_error_baseline_model*100,ndigits=1))       
        
    ## Add values to columns in output table    
    df_output_table['test_data_size']      = data_outer_test_length
    df_output_table['n_hidden_units']      = optimal_regularization_param_ANN
    df_output_table['ANN_test_error']      = col_2
    df_output_table['lambda']              = optimal_regularization_param_logitisctic
    df_output_table['Logistic_test_error'] = col_4
    df_output_table['baseline_test_error'] = col_5
    

## Export as csv
df_output_table.to_csv('Classification_summary_table.csv')


### Statistical Test Evaluation (SETUP II)
if(apply_setup_ii):
    most_common_lambda    = stats.mode(optimal_regularization_param_logitisctic).mode[0].astype('float64')    
    y_true = []
    yhat = []
    
    k = 0
    for train_index,test_index in CV_setup_ii.split(X):
        print('Computing setup II CV K-fold: {0}/{1}..'.format(k+1,K_1))
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        
        X_train_tensor = torch.tensor(X[train_index,:], dtype=torch.float)
        y_train_tensor = torch.tensor(y[train_index], dtype=torch.float)
        X_test_tensor = torch.tensor(X[test_index,:], dtype=torch.float)
        y_test_tensor = torch.tensor(y[test_index], dtype=torch.uint8)
        
        model_baseline = stats.mode(y_train).mode[0][0]
        model_logistic = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/most_common_lambda, solver = 'lbfgs').fit(X_train,y_train.squeeze())
        
        yhat_baseline  = np.ones((y_test.shape[0],1))*model_baseline.squeeze()
        yhat_logistic  = model_logistic.predict(X_test).reshape(-1,1) ## use reshape to ensure it is a nested array

        if(apply_KNN):
            most_common_regu_KNN  = stats.mode(optimal_regularization_param_KNN).mode[0].astype('int64')
            model_second          = KNeighborsClassifier(n_neighbors=most_common_regu_KNN, p=dist,                                  metric=metric,
                                    metric_params=metric_params).fit(X_train.squeeze(), y_train.squeeze())
            y_hat_second_model = model_second.predict(X_test.squeeze()).reshape(-1,1) 
            
        if(apply_ANN):
            most_common_regu_ANN  = stats.mode(optimal_regularization_param_ANN).mode[0].astype('float64')
            model_second = lambda: torch.nn.Sequential(
                                    torch.nn.Linear(M, most_common_regu_ANN), #M features to H hidden units
                                    # 1st transfer function, either Tanh or ReLU:
                                    #torch.nn.ReLU(), 
                                    torch.nn.Tanh(),   
                                    torch.nn.Linear(most_common_regu_ANN, 1), # H hidden units to 1 output neuron
                                    torch.nn.Sigmoid() # final tranfer function
                                    )
        
            ## Run optimization
            net, final_loss, learning_curve = train_neural_net(model_second,
                                               loss_fn,
                                               X=X_train_tensor,
                                               y=y_train_tensor,
                                               n_replicates=n_rep_ann,
                                               max_iter=max_iter)
            # Determine estimated class labels for test set
            y_sigmoid           = net(X_test_tensor) # activation of final note, i.e. prediction of network
            y_hat_second_model = y_sigmoid > .5 # threshold output of sigmoidal function
            
        ## Add true classes and store estimated classes    
        y_true.append(y_test)
        yhat.append( np.concatenate([yhat_baseline, yhat_logistic,y_hat_second_model], axis=1) )
        
        ## Compute the r test size and store it
        r_baseline_vs_logistic.append( np.mean( np.abs( yhat_baseline-y_test ) ** loss_in_r_function - np.abs( yhat_logistic-y_test) ** loss_in_r_function ) )
        r_baseline_vs_sec_model.append( np.mean( np.abs( yhat_baseline-y_test ) ** loss_in_r_function - np.abs( y_hat_second_model-y_test) ** loss_in_r_function ) )
        r_sec_model_vs_logistic.append( np.mean( np.abs( y_hat_second_model-y_test ) ** loss_in_r_function - np.abs( yhat_logistic-y_test) ** loss_in_r_function ) )
        
        ## add to counter
        k += 1


    ## Baseline vs logistic regression    
    p_setupII_base_vs_log, CI_setupII_base_vs_log = correlated_ttest(r_baseline_vs_logistic, rho_t_test, alpha=alpha_t_test)
    
    ## Baseline vs 2nd model    
    p_setupII_base_vs_sec_model, CI_setupII_base_vs_sec_model = correlated_ttest(r_baseline_vs_sec_model, rho_t_test, alpha=alpha_t_test)
    
    ## Logistic regression vs 2nd model    
    p_setupII_log_vs_sec_model, CI_setupII_log_vs_sec_model = correlated_ttest(r_sec_model_vs_logistic, rho_t_test, alpha=alpha_t_test)

    ## Create output table for statistic tests
    df_output_table_statistics = pd.DataFrame(np.ones((3,5)), columns = ['H_0','p_value','CI_lower','CI_upper','conclusion'])
    df_output_table_statistics[['H_0']] = ['err_baseline-err_logistic=0','err_2nd_model-err_logistic=0','baseline_model_err-err_2nd_model_err=0']
    df_output_table_statistics[['p_value']]         = [p_setupII_base_vs_log,p_setupII_log_vs_sec_model,p_setupII_base_vs_sec_model]
    df_output_table_statistics[['CI_lower']]        = [CI_setupII_base_vs_log[0],CI_setupII_log_vs_sec_model[0],CI_setupII_base_vs_sec_model[0]]
    df_output_table_statistics[['CI_upper']]        = [CI_setupII_base_vs_log[1],CI_setupII_log_vs_sec_model[1],CI_setupII_base_vs_sec_model[1]]
    rejected_null                                   = (df_output_table_statistics.loc[:,'p_value']<alpha_t_test)
    df_output_table_statistics.loc[rejected_null,'conclusion']   = 'H_0 rejected'
    df_output_table_statistics.loc[~rejected_null,'conclusion']  = 'H_0 not rejected'
    df_output_table_statistics                      = df_output_table_statistics.set_index('H_0')
    
    ## Export df as csv
    df_output_table_statistics.to_csv('assignment_2_statistic_test.csv',encoding='UTF-8')