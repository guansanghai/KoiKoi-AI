#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:31:17 2021

@author: guansanghai
"""

import PySimpleGUI as sg # pip install pysimplegui
import sys


path_card = 'resource/cardpng/'
path_card_dark = 'resource/cardpngdark/'
path_card_light = 'resource/cardpnglight/'
path_card_small = 'resource/cardpngsmall/'
path_card_small_dark = 'resource/cardpngsmalldark/'
path_card_small_light = 'resource/cardpngsmalllight/'


def CardClassify(cardList):
    brightList = [[1,1],[3,1],[8,1],[11,1],[12,1]]
    seedList = [[2,1],[4,1],[5,1],[6,1],[7,1],[8,2],[9,1],[10,1],[11,2]]
    ribbonList = [[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[9,2],[10,2],[11,3]]
    cardBright, cardSeed, cardRibbon, cardDross = [], [], [], []
    for card in cardList:
        if card == [9,1]:
            cardSeed.append(card)
            cardDross.append(card)
        elif card in brightList:
            cardBright.append(card)
        elif card in seedList:
            cardSeed.append(card)
        elif card in ribbonList:
            cardRibbon.append(card)
        else:
            cardDross.append(card)
    return (cardBright, cardSeed, cardRibbon, cardDross)


def InitGUI():
    sg.theme('Material1')
    
    layoutScoreBoard = [[sg.Text('Round',font=('Helvetica',20),pad=((2,2),(0,0)))],
                        [sg.Text('12 / 12',font=('Helvetica',25),pad=((2,2),(0,3)),key='RoundCounter')],
                        [sg.Text('            ',font=('Helvetica',12),key='gameNum')],
                        [sg.T('')],
                        [sg.Text('Player2Name',font=('Helvetica',20),key='opName')],
                        [sg.Text('30 Points',font=('Helvetica',18),key='opPoints')],
                        [sg.Text('            ',font=('Helvetica',12),key='opDealer')],
                        [sg.T('')],
                        [sg.T(''), sg.Button(image_filename=path_card+'0-0.png',key='PileCard')],
                        [sg.T('')],
                        [sg.Text('Player1Name',font=('Helvetica',20),key='myName')],
                        [sg.Text('30 Points',font=('Helvetica',18),key='myPoints')],
                        [sg.Text('            ',font=('Helvetica',12),key='myDealer')],
                        [sg.T('')],
                        [sg.T('',size=(3,1),key='PointsRound'+str(i)) for i in[1,2,3]],
                        [sg.T('',size=(3,1),key='PointsRound'+str(i)) for i in[4,5,6]],
                        [sg.T('',size=(3,1),key='PointsRound'+str(i)) for i in[7,8,9]],
                        [sg.T('',size=(3,1),key='PointsRound'+str(i)) for i in[10,11,12]],
                        [sg.T('')],
                        [sg.T('')],
                        [sg.Button('Quit',size=(10,1))]
                        ]
    
    
    layoutOpCollectedCardsBrights = [[sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)))],
                                     [sg.Image(path_card_small+'null.png',pad=((0,0),(0,8)),key='OpBrights'+str(i)) for i in range(1,6)]]
    layoutOpCollectedCardsSeeds = [[sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)),key='OpSeeds'+str(i)) for i in range(6,11)],
                                   [sg.Image(path_card_small+'null.png',pad=((0,0),(0,8)),key='OpSeeds'+str(i)) for i in range(1,6)]]
    layoutOpCollectedCardsRibbons = [[sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)),key='OpRibbons'+str(i)) for i in range(6,11)],
                                     [sg.Image(path_card_small+'null.png',pad=((0,0),(0,8)),key='OpRibbons'+str(i)) for i in range(1,6)]]
    layoutOpCollectedCardsDross = [[sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)),key='OpDross'+str(i)) for i in [6,7,8,9,10,16,17,18,19,20,24,25,26]],
                                     [sg.Image(path_card_small+'null.png',pad=((0,0),(0,8)),key='OpDross'+str(i)) for i in [1,2,3,4,5,11,12,13,14,15,21,22,23]]]
    
    layoutOpCollectedCards = [[sg.Column(layoutOpCollectedCardsBrights),
                               sg.Column(layoutOpCollectedCardsSeeds),
                               sg.Column(layoutOpCollectedCardsRibbons),
                               sg.Column(layoutOpCollectedCardsDross)]];
    
    
    layoutOpHandCards = [[sg.Button(image_filename=path_card+'0-0.png',key='OpHand'+str(i)) for i in range(1,9)]]
    
    layoutBoardCards = [[sg.T('')],
                        [sg.Button(image_filename=path_card+'null.png',key='Board'+str(i)) for i in [1,3,5,7,9,11,13,15]],
                        [sg.Button(image_filename=path_card+'null.png',key='Board'+str(i)) for i in [2,4,6,8,10,12,14,16]],
                        [sg.T('')]]
    
    layoutMyHandCards = [[sg.Button(image_filename=path_card+'0-0.png',key='MyHand'+str(i)) for i in range(1,9)]]
    
    
    layoutMyCollectedCardsBrights = [[sg.Image(path_card_small+'null.png',pad=((0,0),(8,0)),key='MyBrights'+str(i)) for i in range(1,6)],
                                     [sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)))]]
    layoutMyCollectedCardsSeeds = [[sg.Image(path_card_small+'null.png',pad=((0,0),(8,0)),key='MySeeds'+str(i)) for i in range(1,6)],
                                   [sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)),key='MySeeds'+str(i)) for i in range(6,11)]]
    layoutMyCollectedCardsRibbons = [[sg.Image(path_card_small+'null.png',pad=((0,0),(8,0)),key='MyRibbons'+str(i)) for i in range(1,6)],
                                     [sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)),key='MyRibbons'+str(i)) for i in range(6,11)]]
    layoutMyCollectedCardsDross = [[sg.Image(path_card_small+'null.png',pad=((0,0),(8,0)),key='MyDross'+str(i)) for i in [1,2,3,4,5,11,12,13,14,15,21,22,23]],
                                     [sg.Image(path_card_small+'null.png',pad=((0,0),(0,0)),key='MyDross'+str(i)) for i in [6,7,8,9,10,16,17,18,19,20,24,25,26]]]
    
    
    layoutMyCollectedCards = [[sg.Column(layoutMyCollectedCardsBrights),
                               sg.Column(layoutMyCollectedCardsSeeds),
                               sg.Column(layoutMyCollectedCardsRibbons),
                               sg.Column(layoutMyCollectedCardsDross)]];
    
    
    layoutOpYakus = [[sg.Text('',size=(16,1),key='OpYaku'+str(i)), sg.Text('',size=(2,1),key='OpYakuPt'+str(i))] for i in range(1,11)]
    layoutHint = [[sg.Text('',size=(17,1),key='Hint', text_color='blue')]]
    layoutMyYakus = [[sg.Text('',size=(16,1),key='MyYaku'+str(i)), sg.Text('',size=(2,1),key='MyYakuPt'+str(i))] for i in range(1,11)]
    
    layoutBoard = [[sg.Column(layoutOpHandCards + layoutBoardCards + layoutMyHandCards), sg.Column(layoutOpYakus + layoutHint + layoutMyYakus)]]
    
    layout = [[sg.Column(layoutScoreBoard), sg.Column(layoutOpCollectedCards + layoutBoard + layoutMyCollectedCards)]]
    
    window = sg.Window('Koi-Koi',layout, finalize=True)
    
    window.bind("<Button-1>", 'Any Click')
    for i in range(1,9):
        window['MyHand'+str(i)].bind("<Enter>",'-Enter')
        window['MyHand'+str(i)].bind("<Leave>",'-Leave')   
        
    return window

               
def UpdateGameStatusGUI(window, game_state):
    round_state = game_state.round_state
    rounds, numRound = game_state.round, game_state.round_total
    player1Name, player2Name = game_state.player_name[1], game_state.player_name[2]
    player1Pts, player2Pts = game_state.point[1], game_state.point[2]
    dealer = round_state.dealer
    game = game_state.log
    
    window['RoundCounter'].update(str(rounds)+' / '+str(numRound))
    window['gameNum'].update('  ')
    window['myName'].update(player1Name)
    window['opName'].update(player2Name)
    window['myPoints'].update(str(player1Pts)+' Points')
    window['opPoints'].update(str(player2Pts)+' Points')

    if dealer == 1:
        window['myDealer'].update('Dealer')
        window['opDealer'].update('      ')
    else:
        window['myDealer'].update('      ')
        window['opDealer'].update('Dealer')

    for i in range(1,rounds):
        window['PointsRound'+str(i)].update(game['record']['round'+str(i)]['basic']['player1RoundPts'])
        
    return window


def UpdateTurnPlayer(window, game_state):
    player1Name, player2Name = game_state.player_name[1], game_state.player_name[2]
    if game_state.round_state.turn_player == 1:
        window['myName'].update(player1Name,text_color='blue')
        window['opName'].update(player2Name,text_color='black')
    else:
        window['myName'].update(player1Name,text_color='black')
        window['opName'].update(player2Name,text_color='blue')
    
    return window



def ClearBoardGUI(window):
    for pre in ['My','Op']:
        for i in range(1,6):
            window[pre+'Brights'+str(i)].update(path_card_small+'null.png')
        for i in range(1,11):
            window[pre+'Seeds'+str(i)].update(path_card_small+'null.png')
        for i in range(1,11):
            window[pre+'Ribbons'+str(i)].update(path_card_small+'null.png')
        for i in range(1,27):
            window[pre+'Dross'+str(i)].update(path_card_small+'null.png')
    
    for pre in ['My','Op']:
        for i in range(1,9):
            window[pre+'Hand'+str(i)].update(image_filename=path_card+'null.png',visible=True)
    
    for pre in ['My','Op']:
        for i in range(1,11):
            window[pre+'Yaku'+str(i)].update('')
            window[pre+'YakuPt'+str(i)].update('')

    return window


def UpdateCardAndYaku(window, game_state):
    window = UpdateHandCardsGUI(window, game_state)
    window = UpdateBoardCardsGUI(window, game_state)
    window = UpdateCollectCardsGUI(window, game_state)
    window = UpdatePileCardGUI(window, game_state)
    window = UpdateYakuGUI(window, game_state)
    
    return window
    

def UpdateCollectCardsGUI(window, game_state):
    round_state = game_state.round_state
    for (pre, CollectCard) in [('My',round_state.pile[1]),('Op',round_state.pile[2])]:
        cardBright, cardSeed, cardRibbon, cardDross = CardClassify(CollectCard)
        for i in range(1,len(cardBright)+1):
            window[pre+'Brights'+str(i)].update(path_card_small+str(cardBright[i-1][0])+'-'+str(cardBright[i-1][1])+'.png')
        for i in range(1,len(cardSeed)+1):
            window[pre+'Seeds'+str(i)].update(path_card_small+str(cardSeed[i-1][0])+'-'+str(cardSeed[i-1][1])+'.png')
        for i in range(1,len(cardRibbon)+1):
            window[pre+'Ribbons'+str(i)].update(path_card_small+str(cardRibbon[i-1][0])+'-'+str(cardRibbon[i-1][1])+'.png')
        for i in range(1,len(cardDross)+1):
            window[pre+'Dross'+str(i)].update(path_card_small+str(cardDross[i-1][0])+'-'+str(cardDross[i-1][1])+'.png')
    
    return window


def UpdateCollectCardsHighlightGUI(window, game_state, card):
    round_state = game_state.round_state
    for (pre, CollectCard) in [('My',round_state.pile[1]),('Op',round_state.pile[2])]:
        cardBright, cardSeed, cardRibbon, cardDross = CardClassify(CollectCard)
        for i in range(1,len(cardBright)+1):
            if cardBright[i-1][0] == card[0]:
                window[pre+'Brights'+str(i)].update(path_card_small_dark+str(cardBright[i-1][0])+'-'+str(cardBright[i-1][1])+'.png')
        for i in range(1,len(cardSeed)+1):
            if cardSeed[i-1][0] == card[0]:
                window[pre+'Seeds'+str(i)].update(path_card_small_dark+str(cardSeed[i-1][0])+'-'+str(cardSeed[i-1][1])+'.png')
        for i in range(1,len(cardRibbon)+1):
            if cardRibbon[i-1][0] == card[0]:
                window[pre+'Ribbons'+str(i)].update(path_card_small_dark+str(cardRibbon[i-1][0])+'-'+str(cardRibbon[i-1][1])+'.png')
        for i in range(1,len(cardDross)+1):
            if cardDross[i-1][0] == card[0]:
                window[pre+'Dross'+str(i)].update(path_card_small_dark+str(cardDross[i-1][0])+'-'+str(cardDross[i-1][1])+'.png')
        
    return window
    

def UpdateHandCardsGUI(window, game_state):
    round_state = game_state.round_state
    myCardList, opCardList  = round_state.hand[1], round_state.hand[2]
    boardCard = round_state.field_slot
    
    ind = [boardCard[i][0] for i in range(0,16)]
    for i in range(1,len(myCardList)+1):
        if myCardList[i-1][0] in ind:
            window['MyHand'+str(i)].update(image_filename=path_card+str(myCardList[i-1][0])+'-'+str(myCardList[i-1][1])+'.png',visible=True)            
        else:
            window['MyHand'+str(i)].update(image_filename=path_card_dark+str(myCardList[i-1][0])+'-'+str(myCardList[i-1][1])+'.png',visible=True)
    for i in range(len(myCardList)+1,9):
        window['MyHand'+str(i)].update(image_filename=path_card+'null.png',visible=True)
    for i in range(1,len(opCardList)+1):
        window['OpHand'+str(i)].update(image_filename=path_card+'0-0.png',visible=True)
    for i in range(len(opCardList)+1,9):
        window['OpHand'+str(i)].update(image_filename=path_card+'null.png',visible=True)
    return window


def UpdateMyDiscardCardGUI(window, game_state):
    round_state = game_state.round_state
    card = round_state.show[0]
    cardList = round_state.hand[1]
    for i in range(1,len(cardList)+1):
        if cardList[i-1] == card:
            window['MyHand'+str(i)].update(image_filename=path_card+str(cardList[i-1][0])+'-'+str(cardList[i-1][1])+'.png')            
        else:
            window['MyHand'+str(i)].update(image_filename=path_card_dark+str(cardList[i-1][0])+'-'+str(cardList[i-1][1])+'.png')
    for i in range(len(cardList)+1,9):
        window['MyHand'+str(i)].update(image_filename=path_card+'null.png')
    
    return window


def UpdateOpDiscardCardGUI(window, game_state):
    round_state = game_state.round_state
    card = round_state.show[0]
    window['OpHand1'].update(image_filename=path_card+str(card[0])+'-'+str(card[1])+'.png')            
    return window


def UpdateBoardCardsGUI(window, game_state):
    round_state = game_state.round_state
    boardCard = round_state.field_slot
    for i in range(1,17):
        if boardCard[i-1] == [0,0]:
            window['Board'+str(i)].update(image_filename=path_card+'null.png')
        else:
            window['Board'+str(i)].update(image_filename=path_card+str(boardCard[i-1][0])+'-'+str(boardCard[i-1][1])+'.png')
    return window


def UpdateBoardCardsHighlightGUI(window, game_state, card):
    round_state = game_state.round_state
    boardCard = round_state.field_slot
    for i in range(1,17):
        if boardCard[i-1][0] == card[0]:
            window['Board'+str(i)].update(image_filename=path_card+str(boardCard[i-1][0])+'-'+str(boardCard[i-1][1])+'.png')
        elif boardCard[i-1] != [0,0]:
            window['Board'+str(i)].update(image_filename=path_card_dark+str(boardCard[i-1][0])+'-'+str(boardCard[i-1][1])+'.png')
    return window


def UpdateYakuGUI(window, game_state):
    round_state = game_state.round_state
    for (pre, yakuList, roundPts) in [('My',round_state.yaku(1),round_state.yaku_point(1)),('Op',round_state.yaku(2),round_state.yaku_point(2))]:
        if len(yakuList) >= 10:
            window[pre+'Yaku1'].update('Too Many Yakus')
            window[pre+'Yaku2'].update('--------TOTAL--------')
            window[pre+'YakuPt2'].update(str(roundPts))
            return window
        for i in range(1,len(yakuList)+1):
            window[pre+'Yaku'+str(i)].update(yakuList[i-1][1])
            window[pre+'YakuPt'+str(i)].update(str(yakuList[i-1][2]))
            if yakuList[i-1][0] == 16 and yakuList[i-1][2] >= 4:
                window[pre+'YakuPt'+str(i)].update('x'+str(yakuList[i-1][2]-2))            
        if len(yakuList) > 0:
            window[pre+'Yaku'+str(len(yakuList)+1)].update('--------TOTAL--------')
            window[pre+'YakuPt'+str(len(yakuList)+1)].update(str(roundPts))
    return window


def UpdatePileCardGUI(window, game_state):
    window['PileCard'].update(image_filename=path_card+'0-0.png')
    return window


def ShowPileCardGUI(window, game_state):
    round_state = game_state.round_state
    card = round_state.show[0]
    window['PileCard'].update(image_filename=path_card+str(card[0])+'-'+str(card[1])+'.png')
    return window


def WaitDiscardGUI(window, game_state):
    round_state = game_state.round_state
    myHandCard = round_state.hand[1]
    window['Hint'].update('-> Select a Hand Card')
    while True:
        event, values = window.read() 
        if event in ['MyHand'+str(i)+'-Enter' for i in range(1,len(myHandCard)+1)]:
            window = UpdateBoardCardsHighlightGUI(window, game_state, myHandCard[int(event[6])-1])
            window = UpdateCollectCardsHighlightGUI(window, game_state, myHandCard[int(event[6])-1])
        if event in ['MyHand'+str(i)+'-Leave' for i in range(1,len(myHandCard)+1)]:
            window = UpdateBoardCardsGUI(window, game_state)
            window = UpdateCollectCardsGUI(window, game_state)
        if event in ['MyHand'+str(i) for i in range(1,len(myHandCard)+1)]:
            discardCardInd = int(event[6])-1
            window = UpdateCollectCardsGUI(window, game_state)
            break
        if event == 'Quit' or event == None:
            window.Close()
            sys.exit(0)
    discardCard = myHandCard[discardCardInd]
    return window, discardCard


def WaitPickGUI(window, game_state):
    round_state = game_state.round_state
    boardCard = round_state.field_slot
    discardCard = round_state.show[0]
    cardToCollectInd = [i+1 for i in range(16) if boardCard[i] in round_state.pairing_card]
    window['Hint'].update('-> Select a Field Card')
    window = UpdateBoardCardsHighlightGUI(window, game_state, discardCard)
    while True:
        event, values = window.read()
        if event in ['Board'+str(i) for i in cardToCollectInd]:
            pickCardInd = int(event[5:]) - 1
            break
        if event == 'Quit' or event == None:
            window.Close()
            sys.exit(0)
    pickCard = boardCard[pickCardInd]
    return window, pickCard


def WaitAnyClick(window):
    window['Hint'].update('-> Click to Continue')
    while True:
        event, values = window.read()
        if event == 'Quit' or event == None:
            window.Close()
            sys.exit(0)        
        elif event == 'Any Click':
            break
        
    return window


def WaitKoiKoi(window):
    window['Hint'].update('-> Koi-Koi?')
    event = sg.popup_yes_no('Koi-Koi?')
    while True:
        if event == 'Yes':
            isKoiKoi = True
            break
        if event == 'No':
            isKoiKoi = False
            break
        if event == None:
            window.Close()
            sys.exit(0)  
    
    return window, isKoiKoi


def ShowOpKoiKoi(window, game_state, action):
    PlayerName = game_state.player_name[game_state.round_state.turn_player]
    if action == True:
        sg.popup(PlayerName+': Koi-Koi', title='Koi-Koi')  
    elif action == False:
        sg.popup(PlayerName+': Stop', title='Koi-Koi')
    
    return window


def ShowRoundOverGUI(window, game_state):
    round_state = game_state.round_state
    player1Name, player2Name = game_state.player_name[1], game_state.player_name[2]
    player1RoundPts, player2RoundPts = round_state.round_point[1], round_state.round_point[2]
    window['Hint'].update('-> Round Over')
    sg.popup(player1Name + ': ' + str(player1RoundPts) + ' '*5 + player2Name + ': ' + str(player2RoundPts), title='Round Over')    
    
    return window


def ShowGameOverGUI(window, game_state):
    player1Name, player2Name = game_state.player_name[1], game_state.player_name[2]
    player1Pts, player2Pts = game_state.point[1], game_state.point[2]
    window['Hint'].update('-> Game Over')
    sg.popup(player1Name + ': ' + str(player1Pts) + ' '*5 + player2Name + ': ' + str(player2Pts), title='Game Over')

    return window


def Close(window):
    window.Close()
    return

