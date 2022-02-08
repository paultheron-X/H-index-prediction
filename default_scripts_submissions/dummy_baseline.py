#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
import networkx as nx

# read training data
df_train = pd.read_csv('train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv('test.csv', dtype={'author': np.int64})
n_test = df_test.shape[0]


# write the predictions to file

pred = df_train.groupby('hindex').count().idxmax().values[0]

df_test["hindex"] = pred 


df_test.loc[:,["author","hindex"]].to_csv('dummy_baseline.csv', index=False)



