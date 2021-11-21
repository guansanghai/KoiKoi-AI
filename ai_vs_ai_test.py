#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:42:30 2021

@author: guansanghai
"""

import os
import torch
import koikoilearn

ai_name_pair = ['RL-WP', 'RL-WP'] # 'RL-Point','RL-WP','SL'
record_path = 'gamerecords_agents/'
game_state_kwargs={'player_name':ai_name_pair,
                   'record_path':record_path,
                   'save_record':True}

if not os.path.isdir(record_path):
    os.mkdir(record_path)

ai_agent = {}
for ii, ai_name in enumerate(ai_name_pair):
    assert ai_name in ['RL-Point','RL-WP','SL']
    if ai_name == 'SL':
        discard_model_path = 'model_agent/discard_sl.pt'
        pick_model_path = 'model_agent/pick_sl.pt'
        koikoi_model_path = 'model_agent/koikoi_sl.pt'
    elif ai_name == 'RL-Point':
        discard_model_path = 'model_agent/discard_rl_point.pt'
        pick_model_path = 'model_agent/pick_rl_point.pt'
        koikoi_model_path = 'model_agent/koikoi_rl_point.pt'
    elif ai_name == 'RL-WP':
        discard_model_path = 'model_agent/discard_rl_wp.pt'
        pick_model_path = 'model_agent/pick_rl_wp.pt'
        koikoi_model_path = 'model_agent/koikoi_rl_wp.pt'
    
    discard_model = torch.load(discard_model_path, map_location=torch.device('cpu'))
    pick_model = torch.load(pick_model_path, map_location=torch.device('cpu'))
    koikoi_model = torch.load(koikoi_model_path, map_location=torch.device('cpu'))
    
    ai_agent[ii+1] = koikoilearn.Agent(discard_model, pick_model, koikoi_model)

arena = koikoilearn.Arena(ai_agent[1], ai_agent[2], game_state_kwargs=game_state_kwargs)
arena.multi_game_test(1)
print('Over')

