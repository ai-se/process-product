#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import pickle

from scipy import stats
import scipy.io
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky
from scipy.io import loadmat


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
# from pyearth import Earth

from src import SMOTE
from src import CFS
from src import metrices_V2 as metrices

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

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


# In[ ]:


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def load_data(project):
    understand_path = 'data/understand_files_all/' + project + '_understand.csv'
    commit_guru_path = 'data/commit_guru/' + project + '.csv'
    understand_df = pd.read_csv(understand_path)
    understand_df = understand_df.dropna(axis = 1,how='all')
    cols_list = understand_df.columns.values.tolist()
    for item in ['Kind', 'Name','commit_hash', 'Bugs']:
        if item in cols_list:
            cols_list.remove(item)
            cols_list.insert(0,item)
    understand_df = understand_df[cols_list]
    commit_guru_df = pd.read_csv(commit_guru_path)
    cols = understand_df.columns.tolist()
    
    commit_guru_df = commit_guru_df.drop(labels = ['parent_hashes','author_name','author_name',
                                                   'author_email','fileschanged','author_date',
                                                   'author_date_unix_timestamp', 'commit_message',
                                                  'classification', 'fix', 'contains_bug','fixes',],axis=1)

    understand_df = understand_df.drop_duplicates(cols[4:len(cols)])
    df = understand_df.merge(commit_guru_df,on='commit_hash')
    cols = df.columns.tolist()
    cols = cols[1:] + [cols[0]]
    df = df[cols]
    for item in ['Kind', 'Name','commit_hash']:
        if item in cols:
            df = df.drop(labels = [item],axis=1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('data/converted/'+ project + '_understand.csv',index=False)
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


# In[ ]:


def run_self(project,projects):
    precision_list = []
    recall_list = []
    pf_list = []
    f1_list = []
    g_list = []
    auc_list = []
    pci_20_list = []
    ifa_list = []
    featue_importance = []
    X_train,y_train = load_data(project)
    df_smote = pd.concat([X_train,y_train],axis = 1)
    df_smote = apply_smote(df_smote)
    y_train = df_smote.Bugs
    X_train = df_smote.drop('Bugs',axis = 1)
    clf =  RandomForestClassifier()
    clf.fit(X_train,y_train)
    importance = 0
    for project in projects:
        try:
            X_test,y_test = load_data(project)
            loc = X_test.CountLineCode
            predicted = clf.predict(X_test)
            abcd = metrices.measures(y_test,predicted,loc)
            pf_list.append(abcd.get_pf())
            recall_list.append(abcd.calculate_recall())
            precision_list.append(abcd.calculate_precision())
            f1_list.append(abcd.calculate_f1_score())
            g_list.append(abcd.get_g_score())
            pci_20_list.append(abcd.get_pci_20())
            ifa_list.append(abcd.get_ifa())
            try:
                auc_list.append(roc_auc_score(y_test, predicted))
            except:
                precision_list.append(0)
        except:
            continue
    return np.nanmedian(recall_list),np.nanmedian(precision_list),np.nanmedian(pf_list),np.nanmedian(f1_list),np.nanmedian(g_list),np.nanmedian(auc_list),np.nanmedian(pci_20_list),np.nanmedian(ifa_list),importance


# In[ ]:


def run_self_CFS(project):
    X,y = load_data(project)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=18)
    loc = X_test.CountLineCode
    df_smote = pd.concat([X_train,y_train],axis = 1)
    df_smote = apply_smote(df_smote)
    df_smote,cols = apply_cfs(df_smote)
    y_train = df_smote.Bugs
    X_train = df_smote.drop('Bugs',axis = 1)
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    importance = clf.feature_importances_
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
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance


# In[ ]:


proj_df = pd.read_csv('projects.csv')
projects = proj_df.repo_name.tolist()


# In[ ]:


def run(projects,all_projects):
    precision_list = {}
    recall_list = {}
    pf_list = {}
    f1_list = {}
    g_list = {}
    auc_list = {}
    pci_20_list = {}
    ifa_list = {}
    featue_importance = {}
    for project in projects[0:10]:
        try:
            if project == '.DS_Store':
                continue
    #         if project != 'org.alloytools.alloy':
    #             continue
    #         if project != 'guice':
    #             continue
            print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
            recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance = run_self(project,all_projects)
#             print(recall,precision,pf,f1,g_score,auc,pci_20,ifa)
            recall_list[project] = recall
            precision_list[project] = precision
            pf_list[project] = pf
            f1_list[project] = f1
            g_list[project] = g_score
            auc_list[project] = auc
            pci_20_list[project] = pci_20
            ifa_list[project] = ifa
            featue_importance[project] = importance
        except Exception as e:
            print(e)
            continue
    final_result = {}
    final_result['precision'] = precision_list
    final_result['recall'] = recall_list
    final_result['pf'] = pf_list
    final_result['f1'] = f1_list
    final_result['g'] = g_list
    final_result['auc'] = auc_list
    final_result['pci_20'] = pci_20_list
    final_result['ifa'] = ifa_list
    return final_result



#projects = projects[0:10]
threads = []
results = {}
results['precision'] = {}
results['recall'] = {}
results['pf'] = {}
results['f1'] = {}
results['g'] = {}
results['auc'] = {}
results['pci_20'] = {}
results['ifa'] = {}
cores = cpu_count()
split_projects = np.array_split(projects, cores*2)
for i in range(cores*2):
    print("starting thread ",i)
    t = ThreadWithReturnValue(target = run, args = [split_projects[i],projects])
    threads.append(t)
for th in threads:
    th.start()
for th in threads:
    response = th.join()
    print(response)
    results['precision'].update(response['precision'])
    results['recall'].update(response['recall'])
    results['pf'].update(response['pf'])
    results['f1'].update(response['f1'])
    results['g'].update(response['g'])
    results['auc'].update(response['auc'])
    results['pci_20'].update(response['pci_20'])
    results['ifa'].update(response['ifa'])


with open('results/Performance/process+product_cross_project.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
