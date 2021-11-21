#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 01:16:43 2020

@author: guansanghai
"""

## A GUI for checking game records

# init GUI layout
import PySimpleGUI as sg # pip install pysimplegui
import os.path
import json
import sys

from gamerecord_replay_func import checkYaku, UpdateBoardCard
import gamerecord_replay_gui as GUI

playerView = 1 # 1 or 2
startRound = 1 # 1 to 8 (if exist)
filename = 'gamerecords_dataset/1.json' # path of saved game record

# Check record exists or not
assert os.path.isfile(filename)
with open(filename, 'r') as f:
    game = json.load(f)
# if game is not over, exit
if game['result']['isOver'] == False:
    print('Game is not over')
    sys.exit(0)  

# Load Data
player1Name = game['info']['player1Name']
player2Name = game['info']['player2Name']
numRound = game['info']['numRound']
player1Pts = game['info']['player1InitPts']
player2Pts = game['info']['player2InitPts']

for realRounds in range(1,9):
    if 'round'+str(realRounds+1) not in game['record']:
        break

for rounds in range(1,startRound):
    player1Pts += game['record']['round'+str(rounds)]['basic']['player1RoundPts']
    player2Pts += game['record']['round'+str(rounds)]['basic']['player2RoundPts']

if player1Pts >= 0 and player2Pts >= 0:
    window = GUI.InitReplayGUI()

for rounds in range(startRound,realRounds+1):
    player1HandCard = game['record']['round'+str(rounds)]['basic']['initHand1']
    player2HandCard = game['record']['round'+str(rounds)]['basic']['initHand2']
    boardCard = game['record']['round'+str(rounds)]['basic']['initBoard']
    boardCard.extend([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
    pileCard = game['record']['round'+str(rounds)]['basic']['initPile']
    dealer = game['record']['round'+str(rounds)]['basic']['Dealer']
    player1RoundPts = game['record']['round'+str(rounds)]['basic']['player1RoundPts']
    player2RoundPts = game['record']['round'+str(rounds)]['basic']['player2RoundPts']
    
    player1CollectCard = []
    player2CollectCard = []
    
    opCollectCard = player2CollectCard if playerView==1 else player1CollectCard
    myCollectCard = player1CollectCard if playerView==1 else player2CollectCard
    opHandCard = player2HandCard if playerView==1 else player1HandCard
    myHandCard = player1HandCard if playerView==1 else player2HandCard
    opKoiKoiNum = 0
    myKoiKoiNum = 0
    
    # update GUI                        
    # score board
    window = GUI.ClearBoardGUI(window)                   
    window['RoundCounter'].update(str(rounds)+' / '+str(numRound))
    window['gameNum'].update('Replay')
    if playerView == 1:
        window['myName'].update(player1Name)
        window['opName'].update(player2Name)
        window['myPoints'].update(str(player1Pts)+' Points')
        window['opPoints'].update(str(player2Pts)+' Points')
    else:
        window['myName'].update(player2Name)
        window['opName'].update(player1Name)
        window['myPoints'].update(str(player2Pts)+' Points')
        window['opPoints'].update(str(player1Pts)+' Points')
        

    if dealer == playerView:
        window['opDealer'].update('')
        window['myDealer'].update('Dealer')
    else:
        window['myDealer'].update('')
        window['opDealer'].update('Dealer')

    for i in range(1,rounds):
        window['PointsRound'+str(i)].update(game['record']['round'+str(i)]['basic']['player'+str(playerView)+'RoundPts'])
    
    for i in range(2-int(dealer==playerView),17,2):
        window['OpenPileCard'+str(i)].update(r'resource/cardpngsmall/'+str(pileCard[-i][0])+'-'+str(pileCard[-i][1])+'.png',visible=True)
    for i in range(1+int(dealer==playerView),17,2):
        window['OpenPileCard'+str(i)].update(r'resource/cardpngsmallgrey/'+str(pileCard[-i][0])+'-'+str(pileCard[-i][1])+'.png',visible=True)
    for i in range(17,25):
        window['OpenPileCard'+str(i)].update(r'resource/cardpngsmalldark/'+str(pileCard[-i][0])+'-'+str(pileCard[-i][1])+'.png',visible=True)
    
    window = GUI.UpdateHandCardsGUI(window, 'My', myHandCard, boardCard)
    window = GUI.UpdateHandCardsGUI(window, 'Op', opHandCard, boardCard)
    window = GUI.UpdateBoardCardsGUI(window, boardCard)
    
    for turns in range (1,17):
        playerInTurn = game['record']['round'+str(rounds)]['turn'+str(turns)]['playerInTurn']
        
        # show discard card
        discardCard = game['record']['round'+str(rounds)]['turn'+str(turns)]['discardCard']
        collectCard = game['record']['round'+str(rounds)]['turn'+str(turns)]['collectCard']
        
        window['Hint'].update('-> Click To Continue')
        while True:
            event, values = window.read()
            if event == 'Save & Quit' or event == None:
                window.Close()
                sys.exit(0)
            if event in ['MyHand'+str(i)+'-Enter' for i in range(1,len(myHandCard)+1)]:
                window = GUI.updateBoardCardsHighlightGUI(window, boardCard, myHandCard[int(event[6])-1])
                window = GUI.UpdateCollectCardsHighlightGUI(window, 'My', myCollectCard, myHandCard[int(event[6])-1])
                window = GUI.UpdateCollectCardsHighlightGUI(window, 'Op', opCollectCard, myHandCard[int(event[6])-1])
            if event in ['OpHand'+str(i)+'-Enter' for i in range(1,len(opHandCard)+1)]:
                window = GUI.updateBoardCardsHighlightGUI(window, boardCard, opHandCard[int(event[6])-1])
                window = GUI.UpdateCollectCardsHighlightGUI(window, 'My', myCollectCard, opHandCard[int(event[6])-1])
                window = GUI.UpdateCollectCardsHighlightGUI(window, 'Op', opCollectCard, opHandCard[int(event[6])-1]) 
            if event in ['MyHand'+str(i)+'-Leave' for i in range(1,len(myHandCard)+1)]:
                window = GUI.UpdateBoardCardsGUI(window, boardCard)
                window = GUI.UpdateCollectCardsGUI(window, 'My', myCollectCard)
                window = GUI.UpdateCollectCardsGUI(window, 'Op', opCollectCard)
            if event in ['OpHand'+str(i)+'-Leave' for i in range(1,len(opHandCard)+1)]:
                window = GUI.UpdateBoardCardsGUI(window, boardCard)
                window = GUI.UpdateCollectCardsGUI(window, 'My', myCollectCard)
                window = GUI.UpdateCollectCardsGUI(window, 'Op', opCollectCard)
            if event == 'Any Click':
                break
        
        window = GUI.updateBoardCardsHighlightGUI(window, boardCard, discardCard)
        if playerInTurn == playerView:
            window = GUI.UpdateDiscardCardGUI(window, 'My', discardCard, myHandCard)  
            myHandCard.remove(discardCard)
            myCollectCard.extend(collectCard)
        else:
            window = GUI.UpdateDiscardCardGUI(window, 'Op', discardCard, opHandCard)  
            opHandCard.remove(discardCard)
            opCollectCard.extend(collectCard)
        boardCard = UpdateBoardCard(boardCard, discardCard, collectCard)
        
        while True:
            event, values = window.read()
            if event == 'Save & Quit' or event == None:
                window.Close()
                sys.exit(0)        
            elif event == 'Any Click':
                break
            
        # show collect card
        window = GUI.UpdateHandCardsGUI(window, 'My', myHandCard, boardCard)
        window = GUI.UpdateHandCardsGUI(window, 'Op', opHandCard, boardCard)
        window = GUI.UpdateBoardCardsGUI(window, boardCard)
        window = GUI.UpdateCollectCardsGUI(window, 'My', myCollectCard)
        window = GUI.UpdateCollectCardsGUI(window, 'Op', opCollectCard)
        
        yakuList, myRoundPts = checkYaku(myCollectCard, myKoiKoiNum)
        window = GUI.UpdateYakuGUI(window, 'My', yakuList, myRoundPts)
        yakuList, opRoundPts = checkYaku(opCollectCard, opKoiKoiNum)
        window = GUI.UpdateYakuGUI(window, 'Op', yakuList, opRoundPts)
        
        while True:
            event, values = window.read()
            if event == 'Save & Quit' or event == None:
                window.Close()
                sys.exit(0)        
            elif event == 'Any Click':
                break        

        # draw pile card
        drawCard = game['record']['round'+str(rounds)]['turn'+str(turns)]['drawCard']
        collectCard = game['record']['round'+str(rounds)]['turn'+str(turns)]['collectCard2']

        # GUI show pile card
        window = GUI.updatePileCardGUI(window, drawCard)
        window = GUI.updateBoardCardsHighlightGUI(window, boardCard, drawCard)
        window['OpenPileCard'+str(turns)].update(r'resource/cardpngsmall/null.png')
        
        while True:
            event, values = window.read()
            if event == 'Save & Quit' or event == None:
                window.Close()
                sys.exit(0)        
            elif event == 'Any Click':
                break
        
        if playerInTurn == playerView:
            myCollectCard.extend(collectCard)
        else:
            opCollectCard.extend(collectCard)
        boardCard = UpdateBoardCard(boardCard, drawCard, collectCard)
        
        # show collect card
        window = GUI.updatePileCardGUI(window, None)
        window = GUI.UpdateBoardCardsGUI(window, boardCard)
        window = GUI.UpdateCollectCardsGUI(window, 'My', myCollectCard)
        window = GUI.UpdateCollectCardsGUI(window, 'Op', opCollectCard)
        
        yakuList, myRoundPts = checkYaku(myCollectCard, myKoiKoiNum)
        window = GUI.UpdateYakuGUI(window, 'My', yakuList, myRoundPts)
        yakuList, opRoundPts = checkYaku(opCollectCard, opKoiKoiNum)
        window = GUI.UpdateYakuGUI(window, 'Op', yakuList, opRoundPts)
        
        isKoiKoi = collectCard = game['record']['round'+str(rounds)]['turn'+str(turns)]['isKoiKoi']
        
        if isKoiKoi == None:
            continue
        
        if isKoiKoi == True:
            if playerInTurn == playerView:
                myKoiKoiNum += 1
            else:
                opKoiKoiNum += 1
            
            yakuList, myRoundPts = checkYaku(myCollectCard, myKoiKoiNum)
            window = GUI.UpdateYakuGUI(window, 'My', yakuList, myRoundPts)
            yakuList, opRoundPts = checkYaku(opCollectCard, opKoiKoiNum)
            window = GUI.UpdateYakuGUI(window, 'Op', yakuList, opRoundPts)
            
            sg.popup('Player'+str(playerInTurn)+': Koi-Koi', title='Koi-Koi')
            
        else:
            sg.popup('Player'+str(playerInTurn)+': Stop', title='Koi-Koi')
            break

    sg.popup(player1Name + ': ' + str(player1RoundPts) + ' ' + player2Name + ': ' + str(player2RoundPts), title='Result')
    player1Pts += game['record']['round'+str(rounds)]['basic']['player1RoundPts']
    player2Pts += game['record']['round'+str(rounds)]['basic']['player2RoundPts']

sg.popup(player1Name + ': ' + str(player1Pts) + ' ' + player2Name + ': ' + str(player2Pts), title='Game Over') 
window.Close()


