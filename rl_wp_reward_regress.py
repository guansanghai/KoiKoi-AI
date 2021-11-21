#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 16:26:22 2021

@author: guansanghai
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns


# Linear Regression
win_prob_mat = np.zeros([2,9,61])

with open('result_wp_sim.pkl','rb') as f:
    result_all = pickle.load(f)
    
for round_num in [1,2,3,4,5,6,7,8]:
    result = [item for item in result_all if item[0]==round_num]
    x, y = [], []
    for round_num, point, dealer, win_num in result:
        x.extend([point] * (win_num[0]//2*2 + win_num[1] +  win_num[2]))
        y.extend([1] * (win_num[0]//2))
        y.extend([0] * (win_num[0]//2))
        y.extend([1] * win_num[1])
        y.extend([0] * win_num[2])
        
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    
    win_prob_model = LogisticRegression()
    win_prob_model.fit(x,y)
    
    p = np.arange(61)
    w = win_prob_model.predict_proba(p.reshape(-1,1))[:,1]
    
    win_prob_mat[1,round_num,:] = w
    win_prob_mat[0,round_num,:] = 1-w[::-1]
    plt.plot(p,w)

plt.show()

with open('win_prob_mat_new.pkl','wb') as f:
    pickle.dump(win_prob_mat,f)


# Plot Curve
'''
with open('win_prob_mat_new.pkl','rb') as f:
    result_all = pickle.load(f)

x = result_all[1,:,:]

rnd = []
pt = []
wr = []

for row in [2,4,6,8]:
    rnd.extend([row for _ in range(1,60)])
    pt.extend([ii for ii in range(1,60)])
    wr.extend(list(x[row,1:60]))

df = pd.DataFrame({'Round':rnd,'Point':pt,'Winning Probability':wr})  
plt.figure(figsize=(5, 3)) 
sns.set_theme(style="whitegrid") 
sns.lineplot(data=df,x='Point',y='Winning Probability',hue='Round',
             palette=sns.color_palette('tab10',4))
plt.show()
'''

