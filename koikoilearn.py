#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:11:56 2021

@author: guansanghai
"""

import random
import numpy as np
from collections import namedtuple
import pickle

import koikoigame


class Agent():
    def __init__(self, discard_model, pick_model, koikoi_model):
        self.model = {'discard':discard_model, 'discard-pick':pick_model, 
                      'draw-pick':pick_model, 'koikoi':koikoi_model}
        for key in self.model.keys():
            self.model[key].eval()

        card_list = [[i+1,j+1] for i in range(12) for j in range(4)]
        self.action_dict = {'discard':card_list, 'discard-pick':card_list, 
                            'draw-pick':card_list, 'koikoi':(False, True)}

    def __predict(self, state, feature, mask):
        output = self.model[state](feature).squeeze(0).detach().numpy()
        output = np.exp(output) * mask
        action_output = self.action_dict[state][output.argmax()]        
        return action_output
    
    def auto_action(self, game_state, use_mask=True):
        action_output = None
        if game_state.round_state.wait_action==True:
            turn_player = game_state.round_state.turn_player
            state = game_state.round_state.state
            feature = game_state.feature_tensor.unsqueeze(0)
            if state == 'discard':
                mask = koikoigame.card_to_multi_hot(game_state.round_state.hand[turn_player])
            elif state in ['discard-pick', 'draw-pick']:
                mask = koikoigame.card_to_multi_hot(game_state.round_state.pairing_card)
            elif state == 'koikoi':
                mask = [1,1]
            action_output = self.__predict(state, feature, np.array(mask))     
        return action_output
    
    def auto_random_action(self, game_state):
        state = game_state.round_state.state
        turn_player = game_state.round_state.turn_player
        action_output = None
        if game_state.round_state.wait_action == True:
            if state == 'discard':
                action_output = random.choice(game_state.round_state.hand[turn_player])
            elif state in ['discard-pick', 'draw-pick']:
                action_output = random.choice(game_state.round_state.pairing_card)
            elif state == 'koikoi':
                action_output = random.choice([True, False])
        return action_output     
        
        

        
        