#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 12:09:26 2021

@author: guansanghai
"""

import numpy as np
import torch
import pickle
from sl_train import get_filename_list
import sys

task_name = sys.argv[1] # 'discard', 'pick', 'koi-koi'
model_dir = 'model_sl'

def get_model(task_name, k_fold, fold):
    path = f'{model_dir}/{task_name}_fold_{k_fold}_{fold}.pt'
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

k_fold = 5
model = {ii:get_model(task_name, k_fold, ii) for ii in range(k_fold)}

dataset_path = f'dataset/{task_name}/'
record_num_list = [ii for ii in range(1,201)]
filename_list = get_filename_list(dataset_path, record_num_list)

predict_result_dict = {}

for ii,filename in enumerate(filename_list):
    with open(dataset_path + filename, 'rb') as f:
        sample = pickle.load(f)
    fold = int(filename.split('_')[0]) % k_fold
    feature = sample['feature'].unsqueeze(0)
    output = model[fold](feature).squeeze(0).detach().numpy()
    output_max = output.argmax()
    filt_output_max = (np.exp(output) * sample['action_mask']).argmax()
    predict_result_dict[filename] = {'equal_result':sample['equal_result'],
                                     'output':output_max,
                                     'filt_output':filt_output_max}
    if (ii+1) % 1000 == 0:
        print(f'{ii+1} samples processed...')    
print(f'All {ii+1} samples processed over!')

n = len(predict_result_dict)
n_rough = 0
n_fine = 0
for _, result in predict_result_dict.items():
    if result['output'] in result['equal_result']:
        n_rough += 1
    if result['filt_output'] in result['equal_result']:
        n_fine += 1
print(f'\n{task_name} model')
print(f'Accuracy: {n_rough}/{n} = {n_rough/n}')
print(f'Fine Accuracy: {n_fine}/{n} = {n_fine/n}')

with open(f'{model_dir}/sl_predict_result_dict_{task_name}.pickle', 'wb') as f:
    pickle.dump(predict_result_dict,f)

