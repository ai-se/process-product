#!/usr/bin/env python
# coding: utf-8



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
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold

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




def apply_smote(df):
    df.reset_index(drop=True,inplace=True)
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




def load_both_data(project,metric):
    understand_path = 'data/understand_files_all/' + project + '_understand.csv'
    understand_df = pd.read_csv(understand_path)
    understand_df = understand_df.dropna(axis = 1,how='all')
    cols_list = understand_df.columns.values.tolist()
    for item in ['Kind', 'Name','commit_hash', 'Bugs']:
        if item in cols_list:
            cols_list.remove(item)
            cols_list.insert(0,item)
    understand_df = understand_df[cols_list]
    cols = understand_df.columns.tolist()
    understand_df = understand_df.drop_duplicates(cols[4:len(cols)])
    understand_df['Name'] = understand_df.Name.str.rsplit('.',1).str[1]
    
    commit_guru_file_level_path = 'data/commit_guru_file/' + project + '.csv'
    commit_guru_file_level_df = pd.read_csv(commit_guru_file_level_path)
    commit_guru_file_level_df['commit_hash'] = commit_guru_file_level_df.commit_hash.str.strip('"')
    commit_guru_file_level_df = commit_guru_file_level_df[commit_guru_file_level_df['file_name'].str.contains('.java')]
    commit_guru_file_level_df['Name'] = commit_guru_file_level_df.file_name.str.rsplit('/',1).str[1].str.split('.').str[0].str.replace('/','.')
    commit_guru_file_level_df = commit_guru_file_level_df.drop('file_name',axis = 1)
    
    
    df = understand_df.merge(commit_guru_file_level_df,how='left',on=['commit_hash','Name'])
    
    
    cols = df.columns.tolist()
    cols.remove('Bugs')
    cols.append('Bugs')
    df = df[cols]
    
    for item in ['Kind', 'Name','commit_hash']:
        if item in cols:
            df = df.drop(labels = [item],axis=1)
#     df.dropna(inplace=True)
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    
    y = df.Bugs
    X = df.drop('Bugs',axis = 1)
    cols = X.columns
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    imp_mean = IterativeImputer(random_state=0)
    X = imp_mean.fit_transform(X)
    X = pd.DataFrame(X,columns = cols)
    
    if metric == 'process':
        X = X[['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
       'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
       'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
       'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr']]
    elif metric == 'product':
        X = X.drop(['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
       'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
       'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
       'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr'],axis = 1)
    else:
        X = X
    return X,y


    




def run_self_k(project,projects,metric):
    precision = []
    recall = []
    pf = []
    f1 = []
    g_score = []
    auc = []
    pci_20 = []
    ifa = []
    importance = []
    X_train,y_train = load_both_data(project,metric)
    df_smote = pd.concat([X_train,y_train],axis = 1)
    df_smote = apply_smote(df_smote)
    y_train = df_smote.Bugs
    X_train = df_smote.drop('Bugs',axis = 1)
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    importance = clf.feature_importances_
    for _project in projects:
        try:
            X_test,y_test = load_both_data(_project,metric)
            if metric == 'process':
                loc = X_test['file_la'] + X_test['file_lt']
            elif metric == 'product':
                loc = X_test.CountLineCode
            else:
                loc = X_test['file_la'] + X_test['file_lt']
            predicted = clf.predict(X_test)
            abcd = metrices.measures(y_test,predicted,loc)
            pf.append(abcd.get_pf())
            recall.append(abcd.calculate_recall())
            precision.append(abcd.calculate_precision())
            f1.append(abcd.calculate_f1_score())
            g_score.append(abcd.get_g_score())
            pci_20.append(abcd.get_pci_20())
            ifa.append(abcd.get_ifa())
            try:
                auc.append(roc_auc_score(y_test, predicted))
            except:
                auc.append(0)
        except Exception as e:
            print('error in test',_project,e)
            continue
    return recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance




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
    for project in projects:
        try:
            if project == '.DS_Store':
                continue
            print("+++++++++++++++++   "  + project + "  +++++++++++++++++")
            recall,precision,pf,f1,g_score,auc,pci_20,ifa,importance = run_self_k(project,all_projects,'process')
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
            print('error in train',project,e)
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
    final_result['featue_importance'] = featue_importance
    return final_result



if __name__ == "__main__":
    proj_df = pd.read_csv('projects.csv')
    # proj_df = proj_df[proj_df['repo_name']== 'environment-dashboard']
    projects = proj_df.repo_name.tolist() 
    projects = projects[150:]
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
    split_projects = np.array_split(projects, cores)
    for i in range(cores):
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
    with open('results/Performance/process_cross_project.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)






