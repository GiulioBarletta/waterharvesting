# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:11:50 2022

@author: Giulio
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from statistics import mean, stdev
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import collections
from collections import abc
collections.MutableMapping = abc.MutableMapping
collections.MutableSet = abc.MutableSet
collections.Callable = abc.Callable
collections.Iterable = abc.Iterable
from neupy.algorithms import RBFKMeans
from neupy.algorithms import GRNN
import matplotlib.pyplot as plt


#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'Harvesting data'
data = pd.read_excel(file_name + '.xlsx', header=0)


#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X and Y
X = data.drop(columns = ['Water_volume'], axis = 1)
Y = data.Water_volume

#Using Built in train test split function in sklearn
bins = np.linspace(Y.min(), Y.max()+0.1, 5)
y_binned = np.digitize(Y, bins)


gbm_scores = []
gbm_score_test = []
gbm_mae_test = []
gbm_rmse_test = []
gbm_best_test = []
gbm_best_pred = []

grnn_scores = []
grnn_score_test = []
grnn_mae_test = []
grnn_rmse_test = []
grnn_best_test = []
grnn_best_pred = []

gbm_cv_best = 0
gbm_r2_best = 0
grnn_cv_best = 0
grnn_r2_best = 0

for i in range(50):
    data_train, data_test = train_test_split(data, test_size = 0.25,
                                                 stratify = y_binned, random_state = i)
    
    #Hacking a scaling but keeping columns names since min_max_scaler does not return a dataframe
    minval = data_train.min()
    minmax = data_train.max() - data_train.min()
    data_train_scaled = (data_train - minval) / minmax
    data_test_scaled = (data_test - minval) / minmax
    
    #Define X and Y
    X_train = data_train_scaled.drop(columns = ['Water_volume'], axis=1)
    Y_train = data_train_scaled.Water_volume
    X_test = data_test_scaled.drop(columns = ['Water_volume'], axis=1)
    Y_test = data_test_scaled.Water_volume
    
        
    #Fit the Decision tree
    gbm = GradientBoostingRegressor(criterion='friedman_mse', n_estimators = 300,
                                    learning_rate = 0.01, max_depth = 2)
    
    
    #Cross Validation (CV) process
    cv_score = cross_val_score(gbm, X_train, Y_train, cv = 5)
    gbm_scores.append(cv_score.mean())
    
    
    #Training final algorithms
    gbm.fit(X_train, Y_train) #Gradient Boosting fitting
    
    
    #Gradient Boosting
    gbm_test = gbm.predict(X_test)

    gbm_score_test.append(metrics.r2_score(Y_test, gbm_test))
    gbm_mae_test.append(metrics.mean_absolute_error(Y_test, gbm_test))
    gbm_rmse_test.append(metrics.mean_squared_error(Y_test, gbm_test, squared = False))
    
    if metrics.r2_score(Y_test, gbm_test) > gbm_r2_best and cv_score.mean() > gbm_cv_best :
        gbm_r2_best = metrics.r2_score(Y_test, gbm_test)
        gbm_cv_best = cv_score.mean()
        gbm_best_test = Y_test
        gbm_best_pred = gbm_test
        minmax_1 = minmax[3]
        minval_1 = minval[3]
    
    # RBF and kmeans clustering by class 
    #Number of prototypes
    #prototypes = int(input("Number of seed points:"))
    prototypes = 53
    
    #Finding cluster centers
    df_cluster = X_train
    df_cluster['Water_volume'] = Y_train #Reproduce original data but only with training values
    rbfk_net = RBFKMeans(n_clusters=prototypes) #Chose number of clusters that you want
    rbfk_net.train(df_cluster, epsilon=1e-5)
    center = pd.DataFrame(rbfk_net.centers)
    
    X_train = X_train.drop(columns = ['Water_volume'], axis = 1)
    
    # Turn the centers into prototypes values needed
    X_prototypes = center.iloc[:, 0:-1]
    Y_prototypes = center.iloc[:, -1] #Y_prototypes is the last column of center since 'Water_volume' is the last feature added to center.
    
    #Train GRNN
    GRNNet = GRNN(std=0.25) #Learn more at http://neupy.com/apidocs/neupy.algorithms.rbfn.grnn.html
    GRNNet.train(X_prototypes, Y_prototypes)
    
    # Cross validataion
    cv_score = cross_val_score(GRNNet, X_train, Y_train, scoring = 'r2', cv = 5)
    grnn_scores.append(cv_score.mean())
    
    grnn_test = GRNNet.predict(X_test)
    grnn_score_test.append(metrics.r2_score(Y_test, grnn_test))
    grnn_mae_test.append(metrics.mean_absolute_error(Y_test, grnn_test))
    grnn_rmse_test.append(metrics.mean_squared_error(Y_test, grnn_test, squared = False))
    
    if metrics.r2_score(Y_test, grnn_test) > grnn_r2_best and cv_score.mean() > grnn_cv_best:
        grnn_r2_best = metrics.r2_score(Y_test, grnn_test)
        grnn_cv_best = cv_score.mean()
        grnn_best_test = Y_test
        grnn_best_pred = grnn_test
        minmax_2 = minmax[3]
        minval_2 = minval[3]
    
    
gbm_best_test = gbm_best_test * minmax_1 + minval_1
gbm_best_pred = gbm_best_pred * minmax_1 + minval_1
grnn_best_test = grnn_best_test * minmax_2 + minval_2
grnn_best_pred = grnn_best_pred * minmax_2 + minval_2
    
    
# print scores

print("Average cross validation score (GBR): {0}".format(mean(gbm_scores).round(2)))
gbm_scores_arr = np.array(gbm_scores) 
print("Standard deviation cross validation score (GBR): {0}".format(np.std(gbm_scores_arr).round(2)))
print("Average test set R^2 (GBR): {0}".format(mean(gbm_score_test).round(2)))
gbm_score_test_arr = np.array(gbm_score_test) 
print("Standard deviation R^2 (GBR): {0}".format(np.std(gbm_score_test_arr).round(2)))
print("Average test set MAE (GBR): {0}".format(mean(gbm_mae_test).round(2)))
print("Average test set RMSE (GBR): {0}".format(mean(gbm_rmse_test).round(2)))
print("")
print("Average cross validation score (GRNN): {0}".format(mean(grnn_scores).round(2)))
grnn_scores_arr = np.array(grnn_scores) 
print("Standard deviation cross validation score (GRNN): {0}".format(np.std(grnn_scores_arr).round(2)))
print("Average test set R^2 (GRNN): {0}".format(mean(grnn_score_test).round(2)))
grnn_score_test_arr = np.array(grnn_score_test) 
print("Standard deviation R^2 (GRNN): {0}".format(np.std(grnn_score_test_arr).round(2)))
print("Average test set MAE (GRNN): {0}".format(mean(grnn_mae_test).round(2)))
print("Average test set RMSE (GRNN): {0}".format(mean(grnn_rmse_test).round(2)))
print("")



# Figures

# VS plot

fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(1,2,1)
plt.scatter(gbm_best_test, gbm_best_pred, c = 'green', 
            edgecolor = 'black')
plt.plot([0,11], [0,11], c = 'black')
plt.xlabel('Actual water harvest', fontsize = 20)
plt.ylabel('Predicted water harvest', fontsize = 20)
plt.title('GBR', fontsize = 20)
plt.annotate('$R^2$ = ' + str(gbm_r2_best.round(2)), xy=(0.03, 0.92), 
             xycoords='axes fraction', fontsize = 16)
plt.annotate('CV score = ' + str(gbm_cv_best.round(2)), xy=(0.03, 0.87), 
             xycoords='axes fraction', fontsize = 16)

ax = fig.add_subplot(1,2,2)
plt.scatter(grnn_best_test, grnn_best_pred,
            c = 'green', edgecolor = 'black')
plt.plot([0,11], [0,11], c = 'black')
plt.xlabel('Actual water harvest', fontsize = 20)
plt.title('GRNN', fontsize = 20)
plt.annotate('$R^2$ = ' + str(grnn_r2_best.round(2)), xy=(0.03, 0.92), 
             xycoords='axes fraction', fontsize = 16)
plt.annotate('CV score = ' + str(grnn_cv_best.round(2)), xy=(0.03, 0.87), 
             xycoords='axes fraction', fontsize = 16)


# boxplots

plt.rcParams['figure.dpi'] = 300
meanpointprops = dict(marker='o', markeredgecolor='black',
                      markerfacecolor='firebrick')

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 6))
scores = pd.DataFrame({"GBR": gbm_scores, "GRNN": grnn_scores})
ax1 = scores[['GBR', 'GRNN']].plot(kind='box', title='Overall CV score', 
                                          showmeans = True, meanprops=meanpointprops,
                                          ax = axes[0])
ax1.set_ylim([0, 1])
ax1.set_ylabel('Score', fontsize = 14)
ax1.set_xlabel('Method', fontsize = 14)

scores = pd.DataFrame({"GBR": gbm_score_test, "GRNN": grnn_score_test})
ax2 = scores[['GBR', 'GRNN']].plot(kind='box', title='Overall test set $R^2$', 
                                          showmeans = True, meanprops=meanpointprops,
                                          ax = axes[1])
ax2.set_ylim([0, 1])
ax2.set_ylabel('Score', fontsize = 14)
ax2.set_xlabel('Method', fontsize = 14)
plt.show()

