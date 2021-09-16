#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 08:55:08 2021

@author: guansanghai
"""

import koikoigame
import json
import pickle

def save_sample(record_name, game_state, result):
    state = game_state.round_state.state
    folder = 'dataset'
    subfolder = {'discard':'discard', 'discard-pick':'pick',
                 'draw-pick':'pick', 'koikoi':'koikoi'}[state]
    action_type = {'discard':'d', 'discard-pick':'p1',
                   'draw-pick':'p2', 'koikoi':'k'}[state]
    round_num = game_state.round
    turn_16 = game_state.round_state.turn_16
    filename = f'{record_name}_{round_num}_{turn_16}_{action_type}.pickle'
    path = f'{folder}/{subfolder}/{filename}'
    sample = {'feature':game_state.feature_tensor,'result':result}
    with open(path,'wb') as f:
        pickle.dump(sample,f)
    return

def get_action(round_record_dict, turn_16, state):
    turn_key = f'turn{turn_16}'
    if state == 'discard':
        action = round_record_dict[turn_key]['discardCard']
    elif state == 'discard-pick':
        action = round_record_dict[turn_key]['collectCard'][-1]
    elif state == 'draw-pick':
        action = round_record_dict[turn_key]['collectCard2'][-1]
    elif state == 'koikoi':
        action = round_record_dict[turn_key]['isKoiKoi']    
    if state == 'koikoi':
        result = int(action)
    else:
        result = (action[0]-1) * 4 + (action[1]-1)    
    return action, result

def round_replay_transform(round_num, init_point, round_record_dict, record_name):
    init_dealer = round_record_dict['basic']['Dealer']
    game_state = koikoigame.KoiKoiGameState(
        round_num=round_num, init_point=init_point, init_dealer=init_dealer)
    game_state.round_state.hand[1] = sorted(round_record_dict['basic']['initHand1'].copy())
    game_state.round_state.hand[2] = sorted(round_record_dict['basic']['initHand2'].copy())
    game_state.round_state.field_slot = sorted(round_record_dict['basic']['initBoard'].copy()) + [[0,0]] * 10
    game_state.round_state.stock = round_record_dict['basic']['initPile'].copy()
    
    while not game_state.round_state.round_over:
        state = game_state.round_state.state
        turn_16 = game_state.round_state.turn_16
        if game_state.round_state.wait_action:
            action, result = get_action(round_record_dict, turn_16, state)
            save_sample(record_name, game_state, result)
        else:
            action = None
        
        if state == 'discard':
            game_state.round_state.discard(action)
        elif state == 'discard-pick':
            game_state.round_state.discard_pick(action)            
        elif state == 'draw':
            game_state.round_state.draw(action)            
        elif state == 'draw-pick':
            game_state.round_state.draw_pick(action)            
        elif state == 'koikoi':
            game_state.round_state.claim_koikoi(action)
    return

def get_round_info(record_dict):
    round_list, init_point_list_1, init_point_list_2 = [],[],[]
    point_1 = record_dict['info']['player1InitPts']
    point_2 = record_dict['info']['player2InitPts']
    for ii in range(1,9):
        round_key = f'round{ii}'
        if round_key not in record_dict['record']:
            break
        round_list.append(ii)
        init_point_list_1.append(point_1)
        init_point_list_2.append(point_2)
        point_1 += record_dict['record'][round_key]['basic']['player1RoundPts']
        point_2 += record_dict['record'][round_key]['basic']['player2RoundPts']  
    return round_list, init_point_list_1, init_point_list_2

def record_dict_to_samples(record_dict,record_name):
    for round_num, point_1, point_2 in zip(*get_round_info(record_dict)):
        round_record_dict = record_dict['record'][f'round{round_num}']
        round_replay_transform(round_num, [point_1,point_2], round_record_dict,record_name)
    return


if __name__ == '__main__':
    folder = 'gamerecords_dataset'
    filename_list = [str(ii) for ii in range(1,11)]
    for filename in filename_list:
        path = f'{folder}/{filename}.json'
        with open(path,'r') as f: 
            record_dict = json.load(f)
        record_dict_to_samples(record_dict,filename)
        print(f'record {filename} processed!')
