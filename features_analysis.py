# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:48:24 2022

@author: r0814655
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import cross_val_score
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

 

path_features = 'C:\\Belgium test Project\\Features\\'
en_f = os.listdir(path_features)

path_mat_features = 'C:\\Belgium test Project\\Features from MATLAB\\post_pro_input.csv'

path_label = 'C:\\Belgium test Project\\Label\\Label.csv'

features = pd.read_csv(path_features + en_f[0])
Label = pd.read_csv(path_label)

X = features.to_numpy()
Y = Label['Label'].to_numpy()

Tree = tree.DecisionTreeClassifier(max_depth=5,)

Tree_py = Tree.fit(X,Y)

dot_data = tree.export_graphviz(Tree_py, out_file=None, 
                      feature_names=features.columns,   
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
# graph.render('Py_model', view=True)

input_mat = pd.read_csv(path_mat_features)
X_mat = input_mat.drop(columns = 'Label').to_numpy()
Y_mat = input_mat['Label'].to_numpy()

Tree_mat = Tree.fit(X_mat,Y_mat)
dot_data = tree.export_graphviz(Tree, out_file=None, 
                      feature_names=input_mat.drop(columns = 'Label').columns,   
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
# graph.render('Mat_model', view=True)

# clf = RandomForestClassifier()
# scores_rf = cross_val_score(clf, X, Y, cv=5)

# clf_svc = SVC()
# scores_svc = cross_val_score(clf_svc, X, Y, cv=5)

# clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100000)
# scores_sgd = cross_val_score(clf_sgd, X, Y, cv=5)

# new_features = features[['Mean intensity of daughter cell', 'Area of mother cell', 
#                          'Variance of daughter Euclidean distance',
#                          'Mean intensity of daughter cell (phase)']]

# new_features_np = new_features.to_numpy()

# clf = RandomForestClassifier()
# scores_rf_re = cross_val_score(clf, new_features_np, Y, cv=5)

# clf_svc = SVC(kernel = 'rbf', C = 1)
# scores_svc_re = cross_val_score(clf_svc, new_features_np, Y, cv=5)

# clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100000)
# scores_sgd_re = cross_val_score(clf_sgd, new_features_np, Y, cv=5)




