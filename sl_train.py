#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 21:57:57 2021

@author: guansanghai
"""

import torch # 1.8.1
import torch.utils.data as data

import numpy as np

import os
import pickle

from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_save_dir = 'model_sl'

def get_filename_list(dataset_path, record_num_list):
    filename_list = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.pickle') and int(file.split('_')[0]) in record_num_list:
                filename_list.append(file)
    return filename_list

class KoiKoiSLDataset(data.Dataset):
    def __init__(self, dataset_path, record_num_list):
        self.data = []
        filename_list = get_filename_list(dataset_path, record_num_list)
        for ii,filename in enumerate(filename_list):
            with open(dataset_path + filename, 'rb') as f:
                sample = pickle.load(f)
            self.data.append(sample)
            if (ii+1) % 1000 == 0:
                print(f'{ii+1} samples loaded...')
        print(f'All {ii+1} samples loaded over!')
    
    def __getitem__(self, index):
        sample = self.data[index]
        return sample['feature'], sample['result']
        
    def __len__(self):
        return len(self.data)


class KoiKoiSLTrainer():
    def __init__(self, task_name):
        self.task_name = task_name

    def init_dataset(self, dataset_path, k_fold, test_fold, batch_size, record_num=200):
        self.k_fold = k_fold
        self.test_fold = test_fold
        print('Loading train dataset...')
        train_record_index = [ii for ii in range(1,record_num+1) if ii % k_fold != test_fold]
        train_dataset = KoiKoiSLDataset(dataset_path, train_record_index)
        self.train_loader = data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print('Loading test dataset...')
        test_record_index =  [ii for ii in range(1,record_num+1) if ii % k_fold == test_fold]
        test_dataset = KoiKoiSLDataset(dataset_path, test_record_index)
        self.test_loader = data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
        return

    def init_model(self, net_model, transfer_model_path=None, lr=1e-3):
        self.model = net_model().to(device)
        if transfer_model_path is not None:
            trained_model = torch.load(transfer_model_path)
            model_state_dict = self.model.state_dict()
            update_state_dict = {k:v for k,v in trained_model.state_dict().items() \
                if k in model_state_dict.keys()}
            model_state_dict.update(update_state_dict)
            self.model.load_state_dict(model_state_dict)
            print(f'Trained model {transfer_model_path} loaded!')
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)                  
        return

    def train(self, epoch_num):
        best_acc = 0
        for epoch in range(epoch_num):
            acc, loss = self.__forward_prop(self.train_loader, update_model=True)
            print(f'\nEpoch {epoch+1} train over, acc = {acc:.3f}, loss = {loss:.5f}')
            
            acc, loss = self.__forward_prop(self.test_loader, update_model=False)
            print(f'Epoch {epoch+1} test over, acc = {acc:.3f}, loss = {loss:.5f}')

            if acc > best_acc:
                best_acc = acc
                path = f'{model_save_dir}/{self.task_name}_fold_{self.k_fold}_{self.test_fold}.pt'
                torch.save(self.model, path)
                print(f'Model {path} saved!')
        return

    def __forward_prop(self, data_loader, update_model=True):
        def accuracy(output,result):
            return np.sum(output==result) / float(len(output))

        if update_model:
            self.model.train()
        else:
            self.model.eval()

        acc_list, loss_list = [], []
        for step, (feature, result) in enumerate(data_loader):
            output = self.model(feature.to(device))
            loss = self.criterion(output,result.to(device))
            
            if update_model:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            output = output.argmax(dim=1).cpu().detach().numpy()
            result = result.cpu().numpy()
            acc_list.append(accuracy(output,result))
            loss_list.append(loss.item())
            
        return np.mean(acc_list), np.mean(loss_list)


if __name__ == '__main__':
    pass
    
    '''
    task_name = 'discard'
    dataset_path = f'dataset/{task_name}/'
    net_model = {'discard':DiscardModel,'pick':PickModel,'koikoi':KoiKoiModel}[task_name]
    trained_model_path = None

    trainer = KoiKoiSLTrainer(task_name)
    trainer.init_dataset(dataset_path, k_fold=5, test_fold=0, batch_size=512)
    trainer.init_model(net_model, trained_model_path)
    trainer.train(epoch_num=20)
    '''
