#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import math
import pickle

from scipy import stats
import scipy.io
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky
from scipy.io import loadmat

import matlab.engine as engi
import matlab as mat

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pyearth import Earth

from src import SMOTE
from src import CFS
from src import metrices_V2 as metrices

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
from pathlib import Path

import matplotlib.pyplot as plt


# In[83]:


def load_data(project):
    understand_path = 'data/understand_files_all/' + project + '_understand.csv'
    commit_guru_path = 'data/commit_guru/' + project + '.csv'
    commit_guru_df = pd.read_csv(commit_guru_path)
#     print(commit_guru_df.columns)
    commit_guru_df = commit_guru_df.drop(labels = ['parent_hashes','author_name','author_name',
                                                   'author_email','author_date','fileschanged',
                                                   'author_date_unix_timestamp', 'commit_message','commit_hash',
                                                  'classification', 'fix','fixes',],axis=1)

    df = commit_guru_df
    df.rename(columns={"contains_bug": "Bugs"},inplace=True)
    y = df.Bugs
    X = df.drop('Bugs',axis = 1)
    y.fillna(False,inplace=True)
    df['Bugs'] = y
    df.dropna(inplace=True)
    cols = df.columns.tolist()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    y = df.Bugs
    X = df.drop('Bugs',axis = 1)
    cols = X.columns
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    return X,y

def apply_smote(df):
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df

def apply_cfs(df):
        y = df.Bugs.values
        X = df.drop(labels = ['Bugs'],axis = 1)
        X = X.values
        selected_cols = CFS.cfs(X,y)
        cols = df.columns[[selected_cols]].tolist()
        cols.append('Bugs')
        return df[cols],cols


# In[79]:


def run_self(project):
    X,y = load_data(project)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=18)
    loc = X_test['la'] + X_test['lt']
    df_smote = pd.concat([X_train,y_train],axis = 1)
    df_smote = apply_smote(df_smote)
    y_train = df_smote.Bugs
    X_train = df_smote.drop('Bugs',axis = 1)
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    abcd = metrices.measures(y_test,predicted,loc)
    pf = abcd.get_pf()
    recall = abcd.calculate_recall()
    precision = abcd.calculate_precision()
    f1 = abcd.calculate_f1_score()
    g_score = abcd.get_g_score()
    pci_20 = abcd.get_pci_20()
    ifa = abcd.get_ifa()
    try:
        auc = roc_auc_score(y_test, predicted)
    except:
        auc = 0
    print(classification_report(y_test, predicted))
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa


# In[89]:


def run_self_CFS(project):
    X,y = load_data(project)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=18)
    loc = X_test['la'] + X_test['lt']
    df_smote = pd.concat([X_train,y_train],axis = 1)
    df_smote = apply_smote(df_smote)
    df_smote,cols = apply_cfs(df_smote)
    y_train = df_smote.Bugs
    X_train = df_smote.drop('Bugs',axis = 1)
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test[cols[:-1]])
    abcd = metrices.measures(y_test,predicted,loc)
    pf = abcd.get_pf()
    recall = abcd.calculate_recall()
    precision = abcd.calculate_precision()
    f1 = abcd.calculate_f1_score()
    g_score = abcd.get_g_score()
    pci_20 = abcd.get_pci_20()
    ifa = abcd.get_ifa()
    try:
        auc = roc_auc_score(y_test, predicted)
    except:
        auc = 0
    print(classification_report(y_test, predicted))
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa


# In[90]:


proj_df = pd.read_csv('projects.csv')
projects = proj_df.repo_name.tolist()


# In[91]:


precision_list = {}
recall_list = {}
pf_list = {}
f1_list = {}
g_list = {}
auc_list = {}
pci_20_list = {}
ifa_list = {}
for project in projects:
    print(project)
    try:
        if project == '.DS_Store':
            continue
    #     if project != 'guice':
    #         continue
        print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
        recall,precision,pf,f1,g_score,auc,pci_20,ifa = run_self(project)
        recall_list[project] = recall
        precision_list[project] = precision
        pf_list[project] = pf
        f1_list[project] = f1
        g_list[project] = g_score
        auc_list[project] = auc
        pci_20_list[project] = pci_20
        ifa_list[project] = ifa
    except Exception as e:
        print(e)
        continue
    break
final_result = {}
final_result['precision'] = precision_list
final_result['recall'] = recall_list
final_result['pf'] = pf_list
final_result['f1'] = f1_list
final_result['g'] = g_list
final_result['auc'] = auc_list
final_result['pci_20'] = pci_20_list
final_result['ifa'] = ifa_list
with open('results/Performance/process_1000_CFS.pkl', 'wb') as handle:
    pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)




