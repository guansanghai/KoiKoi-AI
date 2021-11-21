#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:36:30 2020

@author: guansanghai
"""

import PySimpleGUI as sg
from gamerecord_replay_func import CardClassify

def InitReplayGUI():
    sg.theme('Material1')
    
    layoutScoreBoard = [[sg.Text('Round',font=('Helvetica',20),pad=((2,2),(0,0)))],
                        [sg.Text('12 / 12',font=('Helvetica',25),pad=((2,2),(0,3)),key='RoundCounter')],
                        [sg.Text('            ',font=('Helvetica',12),key='gameNum')],
                        [sg.T('')],
                        [sg.Text('Player2Name',font=('Helvetica',20),key='opName')],
                        [sg.Text('30 Points',font=('Helvetica',18),key='opPoints')],
                        [sg.Text('            ',font=('Helvetica',12),key='opDealer')],
                        [sg.T('')],
                        [sg.T(''), sg.Button(image_filename=r'resource/cardpng/0-0.png',key='PileCard')],
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
                        [sg.Button('Save & Quit',size=(10,1))]
                        ]
    
    
    layoutOpCollectedCardsBrights = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)))],
                                     [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,8)),key='OpBrights'+str(i)) for i in range(1,6)]]
    layoutOpCollectedCardsSeeds = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)),key='OpSeeds'+str(i)) for i in range(6,11)],
                                   [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,8)),key='OpSeeds'+str(i)) for i in range(1,6)]]
    layoutOpCollectedCardsRibbons = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)),key='OpRibbons'+str(i)) for i in range(6,11)],
                                     [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,8)),key='OpRibbons'+str(i)) for i in range(1,6)]]
    layoutOpCollectedCardsDross = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)),key='OpDross'+str(i)) for i in [6,7,8,9,10,16,17,18,19,20,24,25,26]],
                                     [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,8)),key='OpDross'+str(i)) for i in [1,2,3,4,5,11,12,13,14,15,21,22,23]]]
    
    layoutOpCollectedCards = [[sg.Column(layoutOpCollectedCardsBrights),
                               sg.Column(layoutOpCollectedCardsSeeds),
                               sg.Column(layoutOpCollectedCardsRibbons),
                               sg.Column(layoutOpCollectedCardsDross)]];
    
    
    layoutOpHandCards = [[sg.Button(image_filename=r'resource/cardpng/0-0.png',key='OpHand'+str(i)) for i in range(1,9)]]
    
    layoutBoardCards = [[sg.T('')],
                        [sg.Button(image_filename=r'resource/cardpng/null.png',key='Board'+str(i)) for i in [1,3,5,7,9,11,13,15]],
                        [sg.Button(image_filename=r'resource/cardpng/null.png',key='Board'+str(i)) for i in [2,4,6,8,10,12,14,16]],
                        [sg.T('')]]
    
    layoutMyHandCards = [[sg.Button(image_filename=r'resource/cardpng/0-0.png',key='MyHand'+str(i)) for i in range(1,9)]]
    
    
    layoutMyCollectedCardsBrights = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(8,0)),key='MyBrights'+str(i)) for i in range(1,6)],
                                     [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)))]]
    layoutMyCollectedCardsSeeds = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(8,0)),key='MySeeds'+str(i)) for i in range(1,6)],
                                   [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)),key='MySeeds'+str(i)) for i in range(6,11)]]
    layoutMyCollectedCardsRibbons = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(8,0)),key='MyRibbons'+str(i)) for i in range(1,6)],
                                     [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)),key='MyRibbons'+str(i)) for i in range(6,11)]]
    layoutMyCollectedCardsDross = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(8,0)),key='MyDross'+str(i)) for i in [1,2,3,4,5,11,12,13,14,15,21,22,23]],
                                     [sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(0,0)),key='MyDross'+str(i)) for i in [6,7,8,9,10,16,17,18,19,20,24,25,26]]]
    
    
    layoutMyCollectedCards = [[sg.Column(layoutMyCollectedCardsBrights),
                               sg.Column(layoutMyCollectedCardsSeeds),
                               sg.Column(layoutMyCollectedCardsRibbons),
                               sg.Column(layoutMyCollectedCardsDross)]];
    
    layoutPileCards = [[sg.Image(r'resource/cardpngsmall/null.png',pad=((0,0),(8,0)),key='OpenPileCard'+str(i)) for i in range(1,25)]]
    
    layoutOpYakus = [[sg.Text('',size=(16,1),key='OpYaku'+str(i)), sg.Text('',size=(2,1),key='OpYakuPt'+str(i))] for i in range(1,11)]
    layoutHint = [[sg.Text('',size=(17,1),key='Hint', text_color='blue')]]
    layoutMyYakus = [[sg.Text('',size=(16,1),key='MyYaku'+str(i)), sg.Text('',size=(2,1),key='MyYakuPt'+str(i))] for i in range(1,11)]
    
    layoutBoard = [[sg.Column(layoutOpHandCards + layoutBoardCards + layoutMyHandCards), sg.Column(layoutOpYakus + layoutHint + layoutMyYakus)]]
    
    layout = [[sg.Column(layoutScoreBoard), sg.Column(layoutOpCollectedCards + layoutBoard + layoutMyCollectedCards + layoutPileCards)]]
    
    window = sg.Window('Koi-Koi',layout, finalize=True)
    
    window.bind("<Button-1>", 'Any Click')
    for i in range(1,9):
        window['MyHand'+str(i)].bind("<Enter>",'-Enter')
        window['MyHand'+str(i)].bind("<Leave>",'-Leave')
        window['OpHand'+str(i)].bind("<Enter>",'-Enter')
        window['OpHand'+str(i)].bind("<Leave>",'-Leave') 
        
    return window

def ClearBoardGUI(window):
    for pre in ['My','Op']:
        for i in range(1,6):
            window[pre+'Brights'+str(i)].update(r'resource/cardpngsmall/null.png')
        for i in range(1,11):
            window[pre+'Seeds'+str(i)].update(r'resource/cardpngsmall/null.png')
        for i in range(1,11):
            window[pre+'Ribbons'+str(i)].update(r'resource/cardpngsmall/null.png')
        for i in range(1,27):
            window[pre+'Dross'+str(i)].update(r'resource/cardpngsmall/null.png')
    
    for pre in ['My','Op']:
        for i in range(1,9):
            window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpng/null.png',visible=True)
    
    for pre in ['My','Op']:
        for i in range(1,11):
            window[pre+'Yaku'+str(i)].update('')
            window[pre+'YakuPt'+str(i)].update('')

    return window

def UpdateCollectCardsGUI(window, pre, CollectCard):
    cardBright, cardSeed, cardRibbon, cardDross = CardClassify(CollectCard)
    for i in range(1,len(cardBright)+1):
        window[pre+'Brights'+str(i)].update(r'resource/cardpngsmall/'+str(cardBright[i-1][0])+'-'+str(cardBright[i-1][1])+'.png')
    for i in range(1,len(cardSeed)+1):
        window[pre+'Seeds'+str(i)].update(r'resource/cardpngsmall/'+str(cardSeed[i-1][0])+'-'+str(cardSeed[i-1][1])+'.png')
    for i in range(1,len(cardRibbon)+1):
        window[pre+'Ribbons'+str(i)].update(r'resource/cardpngsmall/'+str(cardRibbon[i-1][0])+'-'+str(cardRibbon[i-1][1])+'.png')
    for i in range(1,len(cardDross)+1):
        window[pre+'Dross'+str(i)].update(r'resource/cardpngsmall/'+str(cardDross[i-1][0])+'-'+str(cardDross[i-1][1])+'.png')
    
    return window

def UpdateCollectCardsHighlightGUI(window, pre, CollectCard, card):
    cardBright, cardSeed, cardRibbon, cardDross = CardClassify(CollectCard)
    for i in range(1,len(cardBright)+1):
        if cardBright[i-1][0] == card[0]:
            window[pre+'Brights'+str(i)].update(r'resource/cardpngsmalldark/'+str(cardBright[i-1][0])+'-'+str(cardBright[i-1][1])+'.png')
    for i in range(1,len(cardSeed)+1):
        if cardSeed[i-1][0] == card[0]:
            window[pre+'Seeds'+str(i)].update(r'resource/cardpngsmalldark/'+str(cardSeed[i-1][0])+'-'+str(cardSeed[i-1][1])+'.png')
    for i in range(1,len(cardRibbon)+1):
        if cardRibbon[i-1][0] == card[0]:
            window[pre+'Ribbons'+str(i)].update(r'resource/cardpngsmalldark/'+str(cardRibbon[i-1][0])+'-'+str(cardRibbon[i-1][1])+'.png')
    for i in range(1,len(cardDross)+1):
        if cardDross[i-1][0] == card[0]:
            window[pre+'Dross'+str(i)].update(r'resource/cardpngsmalldark/'+str(cardDross[i-1][0])+'-'+str(cardDross[i-1][1])+'.png')
    
    return window
    
def UpdateHandCardsGUI(window, pre, cardList, boardCard):
    ind = [boardCard[i][0] for i in range(0,16)]
    for i in range(1,len(cardList)+1):
        if cardList[i-1][0] in ind:
            window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpng/'+str(cardList[i-1][0])+'-'+str(cardList[i-1][1])+'.png')            
        else:
            window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpngdark/'+str(cardList[i-1][0])+'-'+str(cardList[i-1][1])+'.png')
    for i in range(len(cardList)+1,9):
        window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpng/null.png')
    
    return window

def UpdateDiscardCardGUI(window, pre, card ,cardList):
    for i in range(1,len(cardList)+1):
        if cardList[i-1] == card:
            window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpng/'+str(cardList[i-1][0])+'-'+str(cardList[i-1][1])+'.png')            
        else:
            window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpngdark/'+str(cardList[i-1][0])+'-'+str(cardList[i-1][1])+'.png')
    for i in range(len(cardList)+1,9):
        window[pre+'Hand'+str(i)].update(image_filename=r'resource/cardpng/null.png')
    
    return window

def UpdateBoardCardsGUI(window, boardCard):
    for i in range(1,17):
        if boardCard[i-1] == [0,0]:
            window['Board'+str(i)].update(image_filename=r'resource/cardpng/null.png')
        else:
            window['Board'+str(i)].update(image_filename=r'resource/cardpng/'+str(boardCard[i-1][0])+'-'+str(boardCard[i-1][1])+'.png')
    return window

def updateBoardCardsHighlightGUI(window, boardCard, card):
    for i in range(1,17):
        if boardCard[i-1][0] == card[0]:
            window['Board'+str(i)].update(image_filename=r'resource/cardpng/'+str(boardCard[i-1][0])+'-'+str(boardCard[i-1][1])+'.png')
        elif boardCard[i-1] != [0,0]:
            window['Board'+str(i)].update(image_filename=r'resource/cardpngdark/'+str(boardCard[i-1][0])+'-'+str(boardCard[i-1][1])+'.png')
    return window

def UpdateYakuGUI(window, pre, yakuList, roundPts):
    if len(yakuList) >= 10:
        window[pre+'Yaku1'].update('Too Many Yakus')
        window[pre+'Yaku2'].update('--------TOTAL--------')
        window[pre+'YakuPt2'].update(str(roundPts))
        return window
    for i in range(1,len(yakuList)+1):
        window[pre+'Yaku'+str(i)].update(yakuList[i-1][1])
        window[pre+'YakuPt'+str(i)].update(str(yakuList[i-1][2]))
        if yakuList[i-1][0] == 14 and yakuList[i-1][2] >= 4:
            window[pre+'YakuPt'+str(i)].update('x'+str(yakuList[i-1][2]-2))            
    if len(yakuList) > 0:
        window[pre+'Yaku'+str(len(yakuList)+1)].update('--------TOTAL--------')
        window[pre+'YakuPt'+str(len(yakuList)+1)].update(str(roundPts))
    return window

def updatePileCardGUI(window, card):
    if card == None:
        window['PileCard'].update(image_filename=r'resource/cardpng/0-0.png')
    else:
        window['PileCard'].update(image_filename=r'resource/cardpng/'+str(card[0])+'-'+str(card[1])+'.png')
    return window

