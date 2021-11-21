#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:30:19 2021

@author: guansanghai
"""

import random
import json
import time

import numpy as np
import torch


class DefaultVar():
    DEFAULT_ROUND_TOTAL = 8
    DEFAULT_INIT_POINT = 30


def card_to_multi_hot(card_list):
    card_multi_hot = [0 for i in range(48)]
    for card in card_list:
        card_multi_hot[(card[0]-1)*4+(card[1]-1)] = 1
    return card_multi_hot


class KoiKoiCard():
    crane = {(1,1)}
    curtain = {(3,1)}
    moon = {(8,1)}
    rainman = {(11,1)}
    phoenix = {(12,1)}
    sake = {(9,1)}
    
    light  = {(1,1),(3,1),(8,1),(11,1),(12,1)}
    seed   = {(2,1),(4,1),(5,1),(6,1),(7,1),(8,2),(9,1),(10,1),(11,2)}
    ribbon = {(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(9,2),(10,2),(11,3)}
    dross  = {(1,3),(1,4),(2,3),(2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4),(6,3),(6,4),(7,3),
              (7,4),(8,3),(8,4),(9,3),(9,4),(10,3),(10,4),(11,4),(12,2),(12,3),(12,4),(9,1)}
            
    boar_deer_butterfly = {(6,1),(7,1),(10,1)}
    flower_sake = {(3,1),(9,1)}
    moon_sake = {(8,1),(9,1)}
    red_ribbon = {(1,2),(2,2),(3,2)}
    blue_ribbon = {(6,2),(9,2),(10,2)}
    red_blue_ribbon = {(1,2),(2,2),(3,2),(6,2),(9,2),(10,2)}
    

class KoiKoiRoundStateBase():
    def __init__(self, dealer=None):
        assert dealer in [1,2,None]
        self.hand = {1:[], 2:[]}
        self.pile = {1:[], 2:[]}
        self.field_slot = []
        self.stock = []
        
        self.show = []
        self.collect = []
        
        self.turn_16 = 1
        self.dealer = random.randint(1,2) if dealer==None else dealer
        self.koikoi = {1:[0,0,0,0,0,0,0,0], 2:[0,0,0,0,0,0,0,0]}
        self.winner = None
        self.exhausted = False
        self.log = {}
        self.silence = True
        
        self.turn_point = 0
        
        # action
        self.state = 'init'
        self.wait_action = False        
        self.__deal_card()
    
    def new_round(self):
        self.__init__(dealer=self.winner)
        return
    
    @property
    def turn_player(self):
        return 1 if (self.turn_16+self.dealer)%2==0 else 2
    
    @property
    def idle_player(self):
        return 3-self.turn_player
    
    @property
    def turn_8(self):
        return (self.turn_16+1)//2
    
    @property
    def field(self):
        return sorted([slot for slot in self.field_slot if slot!=[0,0]])
    
    @property
    def unseen_card(self):
        return {1:(self.stock+self.hand[2]), 2:(self.stock+self.hand[1])}
    
    @property
    def pairing_card(self):
        return [card for card in self.field if card[0]==self.show[0][0]]
    
    @property
    def field_collect(self):
        collect_card = self.collect.copy()
        if self.show[0] in collect_card:
            collect_card.remove(self.show[0])
        return collect_card
    
    @property
    def koikoi_num(self):
        return {1:sum(self.koikoi[1]), 2:sum(self.koikoi[2])}
    
    @property
    def round_over(self):
        return self.state == 'round-over'
    
    @property
    def round_point(self):
        # round not over
        if self.winner == None:
            return {1:None, 2:None}
        # round over (turn exhausted)
        elif self.exhausted:
            return {1:1, 2:-1} if self.dealer==1 else {1:-1, 2:1}
        # round over (get yaku and stop)
        elif self.winner == 1:
            return {1:self.yaku_point(1), 2:-self.yaku_point(1)}
        else:
            return {1:-self.yaku_point(2), 2:self.yaku_point(2)}
    
    def __deal_card(self):
        # action
        while True:
            card = [[ii+1,jj+1] for ii in range(12) for jj in range (4)]
            random.shuffle(card)
            self.hand[1] = sorted(card[0:8])
            self.hand[2] = sorted(card[8:16])
            self.field_slot = sorted(card[16:24])+[[0,0] for _ in range(10)]
            self.stock = card[24:]
            # check hand-4 and board-4          
            flag = True
            for suit in range(1,13):
                if 4 in [[card[0] for card in self.hand[1]].count(suit),
                         [card[0] for card in self.hand[2]].count(suit),
                         [card[0] for card in self.field].count(suit)]:
                    flag = False        
            if flag:
                break
        # next state
        self.__write_log()   
        self.state = 'discard'
        self.wait_action = True
        return
    
    def __collect_card(self,card):
        if len(self.pairing_card) == 0:
            self.collect = []
            self.field_slot[self.field_slot.index([0,0])] = self.show[0]
        elif len(self.pairing_card) in [1,3]:
            self.collect = self.show + self.pairing_card
            for paired_card in self.pairing_card:
                self.field_slot[self.field_slot.index(paired_card)] = [0,0]
            self.pile[self.turn_player].extend(self.collect)
        else:
            self.collect = self.show + [card]
            self.field_slot[self.field_slot.index(card)] = [0,0]
            self.pile[self.turn_player].extend(self.collect)
        return
    
    def discard(self, card=None):        
        assert self.state == 'discard'
        assert card in self.hand[self.turn_player]
        # action
        self.turn_point = self.yaku_point(self.turn_player)
        ind = self.hand[self.turn_player].index(card)
        self.show = [self.hand[self.turn_player].pop(ind)]
        # next state
        self.__write_log()
        self.state = 'discard-pick'
        self.wait_action = len(self.pairing_card) == 2
        
        return self.state if self.silence else self.__call__()
        
    def discard_pick(self, card=None):
        assert self.state == 'discard-pick'
        assert (card in self.pairing_card) if self.wait_action else (card == None)
        # action
        self.__collect_card(card)
        # next state
        self.__write_log()
        self.state = 'draw'
        self.wait_action = False
        
        return self.state if self.silence else self.__call__()
        
    def draw(self, card=None):
        assert self.state == 'draw'
        # action
        self.show = [self.stock.pop()]
        # next state
        self.__write_log()
        self.state = 'draw-pick'
        self.wait_action = len(self.pairing_card) == 2
        
        return self.state if self.silence else self.__call__()
        
    def draw_pick(self, card=None):
        assert self.state == 'draw-pick'
        assert (card in self.pairing_card) if self.wait_action else (card == None)
        # action
        self.__collect_card(card)
        # next state
        self.__write_log()
        self.state = 'koikoi'
        self.wait_action = (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 < 8)
        
        return self.state if self.silence else self.__call__()
    
    def claim_koikoi(self, is_koikoi=None):
        assert self.state == 'koikoi'
        assert (type(is_koikoi) == bool) if self.wait_action else (is_koikoi == None)
        # action
        if (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 == 8):
            is_koikoi = False
        self.koikoi[self.turn_player][self.turn_8-1] = int(is_koikoi==True)
        self.__write_log(is_koikoi)
        
        # next state
        if is_koikoi == False:
            self.state = 'round-over'
            self.wait_action = False
            # result
            self.winner = self.turn_player
            self.__write_log()
        elif self.turn_16 == 16:
            self.state = 'round-over'
            self.wait_action = False
            self.exhausted = True
            # result
            self.winner = self.dealer
            self.__write_log()
        else:
            self.turn_16 += 1
            self.state = 'discard'
            self.wait_action = True
            
        return self.state if self.silence else self.__call__()

    def yaku(self,player):
        yaku = []
        pile = set([tuple(card) for card in self.pile[player]])
        koikoi_num = self.koikoi_num[player]
        
        num_light = len(pile & KoiKoiCard.light)
        if num_light == 5:
            yaku.append((1,'Five Lights', 10))            
        elif num_light == 4 and (11,1) not in pile:
            yaku.append((2,'Four Lights', 8))            
        elif num_light == 4:
            yaku.append((3,'Rainy Four Lights', 7))            
        elif num_light == 3 and (11,1) not in pile:
            yaku.append((4,'Three Lights', 5))
        
        num_seed = len(pile & KoiKoiCard.seed)
        if KoiKoiCard.boar_deer_butterfly.issubset(pile):
            yaku.append((5,'Boar-Deer-Butterfly', 5))            
        if KoiKoiCard.flower_sake.issubset(pile) and koikoi_num == 0:
            yaku.append((6,'Flower Viewing Sake', 1))            
        elif KoiKoiCard.flower_sake.issubset(pile) and koikoi_num > 0:
            yaku.append((7,'Flower Viewing Sake', 3))
        if KoiKoiCard.moon_sake.issubset(pile) and koikoi_num == 0:
            yaku.append((8,'Moon Viewing Sake', 1))
        elif KoiKoiCard.moon_sake.issubset(pile) and koikoi_num > 0:
            yaku.append((9,'Moon Viewing Sake', 3))
        if num_seed >= 5:
            yaku.append((10,'Tane', num_seed-4))
            
        num_ribbon = len(pile & KoiKoiCard.ribbon)
        if (KoiKoiCard.red_ribbon|KoiKoiCard.blue_ribbon).issubset(pile):
            yaku.append((11,'Red & Blue Ribbons', 10))
        if KoiKoiCard.red_ribbon.issubset(pile):
            yaku.append((12,'Red Ribbons', 5))
        if KoiKoiCard.blue_ribbon.issubset(pile):
            yaku.append((13,'Blue Ribbons', 5))
        if num_ribbon >= 5:
            yaku.append((14,'Tan', num_ribbon-4))
            
        num_dross = len(pile & KoiKoiCard.dross)
        if num_dross >= 10:
            yaku.append((15,'Kasu', num_dross-9))
            
        if koikoi_num > 0:
            yaku.append((16,'Koi-Koi', koikoi_num))
        
        return yaku
    
    def yaku_point(self, player):
        yaku_point = sum([yaku[2] for yaku in self.yaku(player) if yaku[1]!='Koi-Koi'])
        koikoi_num = self.koikoi_num[player]
        if koikoi_num <= 3:
            yaku_point += koikoi_num
        else:
            yaku_point *= koikoi_num - 2
        
        return yaku_point
    
    def __write_log(self, content=None):
        turn = str(self.turn_16)
        if self.state == 'init':
            self.log['basic'] = {}
            self.log['basic']['initHand1'] = self.hand[1].copy()
            self.log['basic']['initHand2'] = self.hand[2].copy()
            self.log['basic']['initBoard'] = self.field.copy()
            self.log['basic']['initPile'] = self.stock.copy()
            self.log['basic']['Dealer'] = self.dealer
        elif self.state == 'discard':
            self.log['turn'+turn] = {}
            self.log['turn'+turn]['playerInTurn'] = self.turn_player
            self.log['turn'+turn]['discardCard'] = self.show[0].copy()
            self.log['turn'+turn]['pairCard'] = self.pairing_card.copy()
        elif self.state == 'discard-pick':
            self.log['turn'+turn]['collectCard'] = self.collect.copy()
        elif self.state == 'draw':
            self.log['turn'+turn]['drawCard'] = self.show[0].copy()
            self.log['turn'+turn]['pairCard2'] = self.pairing_card.copy()
        elif self.state == 'draw-pick':
            self.log['turn'+turn]['collectCard2'] = self.collect.copy()
        elif self.state == 'koikoi':
            self.log['turn'+turn]['isKoiKoi'] = content
        elif self.state == 'round-over':
            self.log['basic']['roundWinner'] = self.winner
            self.log['basic']['player1RoundPts'] = self.round_point[1]
            self.log['basic']['player2RoundPts'] = self.round_point[2]
        return
    
    def __call__(self, view=None):
        view = self.turn_player if view == None else view
        op_view = 3-view
        pile = set([tuple(card) for card in self.pile[view]])
        op_pile = set([tuple(card) for card in self.pile[op_view]])
        
        print('Turn: '+str(self.turn_8)+',  State: '+self.state)
        print('-----------------------------------------------')
        print('Opponent\'s Yaku:')
        print([[yaku[1],yaku[2]] for yaku in self.yaku(op_view)])
        print('Total Point: '+str(self.yaku_point(op_view)))
        print('-----------------------------------------------')
        print('Opponent\'s Pile:')
        print('Light: '+ str(list(op_pile & KoiKoiCard.light)))
        print('Seed: '+ str(list(op_pile & KoiKoiCard.seed)))
        print('Ribbon: '+ str(list(op_pile & KoiKoiCard.ribbon)))
        print('Dross: '+ str(list(op_pile & KoiKoiCard.dross)))
        print('-----------------------------------------------')
        print('Opponent\'s Hand:')
        print([[0,0] for card in self.hand[op_view]])
        print('-----------------------------------------------')
        print('Field:')
        print(self.field)
        print('-----------------------------------------------')
        print('Your Hand:')
        print(self.hand[view])
        print('-----------------------------------------------')
        print('Your Pile:')
        print('Light: '+ str(list(pile & KoiKoiCard.light)))
        print('Seed: '+ str(list(pile & KoiKoiCard.seed)))
        print('Ribbon: '+ str(list(pile & KoiKoiCard.ribbon)))
        print('Dross: '+ str(list(pile & KoiKoiCard.dross)))
        print('-----------------------------------------------')
        print('Your Yaku:')
        print([[yaku[1],yaku[2]] for yaku in self.yaku(view)])
        print('Total Point: '+str(self.yaku_point(view)))
        print('-----------------------------------------------')
        
        if view != self.turn_player:
            print('Opponent\'s turn, waiting action...')
            return
            
        if self.state == 'discard':
            print('Use discard(card) to discard from hand.')
        elif self.state == 'discard-pick':
            print('Discard: '+str(self.show[0]))
            print('Pairing: '+str(self.pairing_card))
            if self.wait_action:
                print('Use discard_pick(card) to pick a pairing field card.')
            else:
                print('Use discard_pick() to continue.')
        elif self.state == 'draw':
            print('Use draw() to draw from stock.')
        elif self.state == 'draw-pick':
            print('Draw: '+str(self.show[0]))
            print('Pairing: '+str(self.pairing_card))
            if self.wait_action:
                print('Use draw_pick(card) to pick a pairing field card.')
            else:
                print('Use draw_pick() to continue.')
        elif self.state == 'koikoi':
            if self.wait_action:
                print('Use claim_koikoi(bool) to koikoi or stop.')
            else:
                print('Use claim_koikoi() to continue.')        
        elif self.state == 'round-over':
            print('Round Over')
            print('Round Point: You '+str(self.round_point[view])+\
                  ', Opponent '+str(self.round_point[op_view]))
        return
        
    
class KoiKoiGameStateBase():
    def __init__(self, round_num=1, round_total=DefaultVar.DEFAULT_ROUND_TOTAL,
                 init_point=[DefaultVar.DEFAULT_INIT_POINT,DefaultVar.DEFAULT_INIT_POINT],
                 init_dealer=None, player_name=['Player1','Player2'], 
                 record_path='', save_record=False):
        
        self.round_total = round_total
        self.init_point = init_point
        self.init_dealer = init_dealer        
        self.player_name = {1:player_name[0], 2:player_name[1]}
        self.record_path = record_path
        self.save_record = save_record
        
        self.round_state = KoiKoiRoundState(dealer=self.init_dealer)
        self.round = round_num
        self.point = {1:init_point[0],2:init_point[1]}
        self.game_over = False
        self.winner = None
        self.log = {}
        self.__init_record()
        
    def new_game(self):
        self.__init__(round_num=1, round_total=self.round_total, 
                      init_point=self.init_point, init_dealer=self.init_dealer,
                      player_name=[self.player_name[1],self.player_name[2]], 
                      record_path=self.record_path, save_record=self.save_record)
        return
        
    def new_round(self):
        assert self.round_state.state == 'round-over'
        self.point[1] = self.point[1]+self.round_state.round_point[1]
        self.point[2] = self.point[2]+self.round_state.round_point[2]
        self.__round_result_record()
        if self.point[1]<=0 or self.point[2]<=0 or self.round==self.round_total:
            self.game_over = True
            self.winner = 1 if self.point[1]>self.point[2] else (2 if self.point[1]<self.point[2] else 0)
            self.__game_result_record()
        else:
            self.round_state.new_round()
            self.round += 1
        return
    
    def __init_record(self):
        self.log['info'] = {'startTime':time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()), 
                            'endTime':None,
                            'player1Name':self.player_name[1],
                            'player2Name':self.player_name[2],
                            'player1InitPts':self.point[1], 
                            'player2InitPts':self.point[2], 
                            'numRound':self.round_total}
        self.log['result'] = {'isOver':False, 'gameWinner':None, 
                              'player1EndPts':None, "player2EndPts":None}
        self.log['save'] = {}
        self.log['record'] = {}
        return
    
    def __round_result_record(self):
        self.log['record']['round'+str(self.round)] = self.round_state.log
        return
    
    def __game_result_record(self):
        self.log['info']['endTime'] = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        self.log['result'] = {'isOver':True, 'gameWinner':self.winner,
                              'player1EndPts':self.point[1], "player2EndPts":self.point[2]}
        if self.save_record == True:
            self.__save_record()
        return
    
    def __save_record(self):
        filename = self.record_path + self.log['info']['startTime'] + ' ' \
            + self.log['info']['player1Name'] + ' vs ' + self.log['info']['player2Name'] +'.json'
        with open(filename, 'w') as f:
            json.dump(self.log, f)
        return    

    def __call__(self):
        print('-----------------------------------------------')
        print('Round: '+str(self.round)+' / '+str(self.round_total))
        print(self.log['info']['player1Name']+': '+str(self.point[1])+', '+\
              self.log['info']['player2Name']+': '+str(self.point[2]))
        if self.game_over:
            print('Game Over')
        print('-----------------------------------------------')
        return


# class KoiKoiKoiRoundState(KoiKoiRoundStateBase):
#     pass
       
class KoiKoiRoundState(KoiKoiRoundStateBase):
    
    def __init__(self, dealer=None):
        super().__init__(dealer)
        self.card_log_dict = {}
        self.__write_card_log_array('init')
    
    def discard(self, card=None): 
        output = super().discard(card)
        self.__write_card_log_array('discard')
        return output
        
    def discard_pick(self, card=None):
        output = super().discard_pick(card)
        self.__write_card_log_array('discard-pick')
        return output
        
    def draw(self, card=None):
        output = super().draw(card)
        self.__write_card_log_array('draw')
        return output
        
    def draw_pick(self, card=None):
        output = super().draw_pick(card)
        self.__write_card_log_array('draw-pick')
        return output
    
    def step(self, action):
        assert self.state in ['discard','discard-pick','draw','draw-pick','koikoi']
        if self.state == 'discard':
            self.discard(action)
        elif self.state == 'discard-pick':
            self.discard_pick(action)            
        elif self.state == 'draw':
            self.draw(action)            
        elif self.state == 'draw-pick':
            self.draw_pick(action)            
        elif self.state == 'koikoi':
            self.claim_koikoi(action)
        return
    
    def __write_card_log_array(self,state):
        
        def card_log_turn_dict():
            ## Card Dynamic Feature by Turn
            cardLogTurn = {}
            cardLogTurn['CardDiscardedAndPaired'] = np.zeros(48)
            cardLogTurn['CardDiscardedAndUnpaired'] = np.zeros(48)
            cardLogTurn['CardPairedByDiscardCollect'] = np.zeros(48)
            cardLogTurn['CardPairedByDiscardUncollect'] = np.zeros(48)
            cardLogTurn['CardDrawnAndPaired'] = np.zeros(48)
            cardLogTurn['CardDrawnAndUnpaired'] = np.zeros(48)
            cardLogTurn['CardPairedByDrawnCollect'] = np.zeros(48)
            cardLogTurn['CardPairedByDrawnUncollect'] = np.zeros(48)
            return cardLogTurn
    
        turn = self.turn_16
        
        if state == 'init':
            for ii in range(1,17):
                self.card_log_dict[ii] = card_log_turn_dict()
                
        elif state == 'discard':
            if self.pairing_card == []:
                self.card_log_dict[turn]['CardDiscardedAndUnpaired'] \
                    = np.array(card_to_multi_hot(self.show))
            else:
                self.card_log_dict[turn]['CardDiscardedAndPaired'] \
                    = np.array(card_to_multi_hot(self.show)) 
            
        elif state == 'discard-pick':
            if self.collect != []:
                pair_by_discard_collect = np.array(card_to_multi_hot(self.field_collect))
                pair_by_discard = np.array(card_to_multi_hot(self.log[f'turn{turn}']['pairCard']))
                self.card_log_dict[turn]['CardPairedByDiscardCollect'] \
                    = pair_by_discard_collect
                self.card_log_dict[turn]['CardPairedByDiscardUncollect'] \
                    = pair_by_discard - pair_by_discard_collect
            
        elif state == 'draw':
            if self.pairing_card == []:
                self.card_log_dict[turn]['CardDrawnAndUnpaired'] \
                    = np.array(card_to_multi_hot(self.show))
            else:
                self.card_log_dict[turn]['CardDrawnAndPaired'] \
                    = np.array(card_to_multi_hot(self.show))
            
        elif state == 'draw-pick':
            if self.collect != []:
                pair_by_discard_collect = np.array(card_to_multi_hot(self.field_collect))
                pair_by_discard = np.array(card_to_multi_hot(self.log[f'turn{turn}']['pairCard2']))
                self.card_log_dict[turn]['CardPairedByDrawnCollect'] \
                    = pair_by_discard_collect
                self.card_log_dict[turn]['CardPairedByDrawnUncollect'] \
                    = pair_by_discard - pair_by_discard_collect
        return
    
    @property
    def action_mask(self):
        if self.state == 'discard':
            mask = card_to_multi_hot(self.hand[self.turn_player])
        elif self.state in ['discard-pick', 'draw-pick']:
            mask = card_to_multi_hot(self.pairing_card)
        elif self.state == 'koikoi':
            mask = [1,1]
        else:
            mask = []
        return np.array(mask)
    
    @property
    def card_log_array(self):
        turn_list = [x for x in range(self.turn_16,0,-1)] + [x for x in range(self.turn_16+1,17)]
        f_array = np.vstack([f for turn in turn_list for _,f in self.card_log_dict[turn].items()])   
        return f_array
    
    @property
    def card_suit_array(self):
        f_array = np.zeros([12,48])
        for ii in range(12):
            f_array[ii,4*ii:4*ii+4] = 1
        return f_array
    
    @property
    def card_init_position_array(self):
        # There was a bug caused by the shallow copy of initHand and initPile in self.log
        # As the result, this fuction in fact got the current hand and unseen card
        # Although the bug has been fixed, for supporting the trained models, keep it as is
        f_dict = {}
        '''
        f_dict['CardInMyHand'] = card_to_multi_hot(
            self.log['basic'][f'initHand{self.turn_player}'])
        f_dict['CardInBoard'] = card_to_multi_hot(self.log['basic']['initBoard'])
        f_dict['CardUnseen'] = card_to_multi_hot(
            self.log['basic'][f'initHand{self.idle_player}'] + self.log['basic']['initPile'])
        '''
        f_dict['CardInMyHand'] = card_to_multi_hot(self.hand[self.turn_player])
        f_dict['CardInBoard'] = card_to_multi_hot(self.log['basic']['initBoard'])
        f_dict['CardUnseen'] = card_to_multi_hot(self.unseen_card[self.turn_player])
        f_array = np.vstack([value for key,value in f_dict.items()])
        return f_array    
    
    @property
    def card_current_position_array(self):
        f_dict = {}
        f_dict['CardInMyHand'] = card_to_multi_hot(self.hand[self.turn_player])
        f_dict['CardInMyCollect'] = card_to_multi_hot(self.pile[self.turn_player])
        f_dict['CardInBoard'] = card_to_multi_hot(self.field)
        # Bug Confirmed, for supporting the trained models, keep it as is
        # f_dict['CardInOpCollect'] = card_to_multi_hot(self.pile[self.idle_player])
        f_dict['CardInOpCollect'] = card_to_multi_hot(self.pile[self.turn_player])
        f_dict['CardUnseen'] = card_to_multi_hot(self.unseen_card[self.turn_player])
        f_array = np.vstack([value for key,value in f_dict.items()])
        return f_array
    
    @property
    def card_pairing_state_array(self):
        f_dict = {}
        if self.state in ['discard-pick','draw-pick']:
            f_dict['CardShowed'] = card_to_multi_hot(self.show)
            f_dict['CardPaired'] = card_to_multi_hot(self.pairing_card)
        else:
            f_dict['CardShowed'] = card_to_multi_hot([])
            f_dict['CardPaired'] = card_to_multi_hot([])
        f_array = np.vstack([value for key,value in f_dict.items()])
        return f_array
    
    @property
    def yaku_status_array(self):
        
        def card_list_to_set(card_list):
            return set([tuple(card) for card in card_list])
            
        card_dict = {
            'Crane':KoiKoiCard.crane,
            'Curtain':KoiKoiCard.curtain,
            'Moon':KoiKoiCard.moon,
            'Rainman':KoiKoiCard.rainman,
            'Phoenix':KoiKoiCard.phoenix,
            'Sake':KoiKoiCard.sake,
            'BoarDeerButterfly':KoiKoiCard.boar_deer_butterfly,
            'Seed':KoiKoiCard.seed,
            'RedRibbon':KoiKoiCard.red_ribbon,
            'BlueRibbon':KoiKoiCard.blue_ribbon,
            'RedAndBlue':KoiKoiCard.red_blue_ribbon,
            'Ribbon':KoiKoiCard.ribbon,
            'Dross':KoiKoiCard.dross}
        
        my_hand_card = card_list_to_set(self.hand[self.turn_player])
        board_card = card_list_to_set(self.field)
        my_collect_card = card_list_to_set(self.pile[self.turn_player])
        op_collect_card = card_list_to_set(self.pile[self.idle_player])
        unseen_card = card_list_to_set(self.hand[self.idle_player]+self.stock)
        
        f_dict = {}
        f_dict['NumMyHand'] = [len(card_set & my_hand_card) for _,card_set in card_dict.items()]
        f_dict['NumBoard'] = [len(card_set & board_card) for _,card_set in card_dict.items()]
        f_dict['NumMyCollect'] = [len(card_set & my_collect_card) for _,card_set in card_dict.items()]
        f_dict['NumOpCollect'] = [len(card_set & op_collect_card) for _,card_set in card_dict.items()]
        f_dict['NumUnseen'] = [len(card_set & unseen_card) for _,card_set in card_dict.items()]
        f_array_card_state = np.concatenate([value for key,value in f_dict.items()])
        f_array_card_state = np.tile(f_array_card_state,(48,1)).T
        f_array_card_key = np.array([card_to_multi_hot(card_set) for _,card_set in card_dict.items()])
        f_array = np.vstack([f_array_card_state,f_array_card_key])
        return f_array
    
    
class KoiKoiGameState(KoiKoiGameStateBase):
    
    @property
    def game_status_array(self):
        
        def feature_tuple(x, power=[0.5,1,2], weight=[1,1,1]):
            return np.abs(float(x)) ** np.array(power) * np.sign(x) * np.array(weight)
        
        def feature_one_hot(pos, feature_length):
            x = np.zeros(feature_length)
            x[pos] = 1
            return x
        
        f_dict = {}
        
        turn_player = self.round_state.turn_player
        idle_player = self.round_state.idle_player
        
        point_diff = self.point[turn_player] - self.point[idle_player]
        f_dict['GamePoint'] = feature_tuple(float(point_diff)/2, [0.5,1,1.5], [1,0.5,0.1])

        f_dict['MyYakuPoint'] = feature_tuple(
            self.round_state.yaku_point(turn_player), [0.5,1,1.5], [1,0.5,0.1])
        f_dict['OpYakuPoint'] = feature_tuple(
            self.round_state.yaku_point(idle_player), [0.5,1,1.5], [1,0.5,0.1])
        
        f_dict['Round'] = feature_one_hot(self.round-1, 8)
        f_dict['Turn'] = feature_one_hot(self.round_state.turn_16-1, 16)
        f_dict['Dealer'] = feature_one_hot(self.round_state.dealer-1, 2)
        
        f_dict['MyKoiKoiNum'] = feature_tuple(
            self.round_state.koikoi_num[turn_player], [1,2], [1,1])
        f_dict['OpKoiKoiNum'] = feature_tuple(
            self.round_state.koikoi_num[idle_player], [1,2], [1,1])
        
        f_dict['MyKoiKoi'] = np.array(self.round_state.koikoi[turn_player])
        f_dict['OpKoiKoi'] = np.array(self.round_state.koikoi[idle_player])
        
        f_array = np.concatenate([value for key,value in f_dict.items()])
        f_array = np.tile(f_array,(48,1)).T
        return f_array
    
    @property
    def reserve_array(self):
        f_array = np.zeros([17,48])
        return f_array
        
    @property
    def feature_tensor(self):
        f = np.vstack([
            self.reserve_array,
            self.game_status_array,
            self.round_state.yaku_status_array,
            self.round_state.card_suit_array,
            self.round_state.card_init_position_array,
            self.round_state.card_current_position_array,
            self.round_state.card_pairing_state_array,
            self.round_state.card_log_array])
        if self.round_state.state == 'koikoi':
            f_token = np.zeros([f.shape[0],2])
            f_token[0:137,:] = f[0:137,0:2]
            f_token[0,0] = 1
            f_token[1,1] = 1
            f = np.hstack([f_token,f])
        f = torch.Tensor(f)
        return f

    