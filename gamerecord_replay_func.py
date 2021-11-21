#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:37:16 2021

@author: guansanghai
"""

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


def checkYaku(CollectCard, KoiKoiNum):
    YakuList = []
    
    # check Brights
    brightList = [[1,1],[3,1],[8,1],[11,1],[12,1]]
    numBrights = 0
    for card in brightList:
        if card in CollectCard:
            numBrights += 1
    if numBrights == 5:
        YakuList.append([1,'Five Lights', 10])
    elif numBrights == 4 and [11,1] not in CollectCard:
        YakuList.append([2,'Four Lights', 8])
    elif numBrights == 4:
        YakuList.append([3,'Rainy Four Lights', 7])
    elif numBrights == 3 and [11,1] not in CollectCard:
        YakuList.append([4,'Three Lights', 5])
    
    # check Seeds
    if [6,1] in CollectCard and [7,1] in CollectCard and [10,1] in CollectCard:
        YakuList.append([5,'Boar-Deer-Butterfly', 5])        
    if [3,1] in CollectCard and [9,1] in CollectCard:
        if KoiKoiNum == 0:
            YakuList.append([6,'Flower Viewing Sake', 1])
        else:
            YakuList.append([7,'Flower Viewing Sake', 3])
    if [8,1] in CollectCard and [9,1] in CollectCard:
        if KoiKoiNum == 0:
            YakuList.append([8,'Moon Viewing Sake', 1])
        else:
            YakuList.append([9,'Moon Viewing Sake', 3])
    seedList = [[2,1],[4,1],[5,1],[6,1],[7,1],[8,2],[9,1],[10,1],[11,2]]
    numSeeds = 0
    for card in seedList:
        if card in CollectCard:
            numSeeds += 1
    if numSeeds >= 5:
        YakuList.append([10,'Tane', numSeeds-4])
    
    # check Ribbons
    if [1,2] in CollectCard and [2,2] in CollectCard and [3,2] in CollectCard and [6,2] in CollectCard and [9,2] in CollectCard and [10,2] in CollectCard:
        YakuList.append([11,'Red & Blue Ribbons', 10])
    if [1,2] in CollectCard and [2,2] in CollectCard and [3,2] in CollectCard:
        YakuList.append([12,'Red Ribbons', 5])
    if [6,2] in CollectCard and [9,2] in CollectCard and [10,2] in CollectCard:
        YakuList.append([13,'Blue Ribbons', 5])
    ribbonList = [[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[9,2],[10,2],[11,3]]
    numRibbons = 0
    for card in ribbonList:
        if card in CollectCard:
            numRibbons += 1
    if numRibbons >= 5:
        YakuList.append([14,'Tan', numRibbons-4])
    
    # check Dross
    drossList = [[1,3],[1,4],[2,3],[2,4],[3,3],[3,4],[4,3],[4,4],[5,3],[5,4],[6,3],[6,4],
                 [7,3],[7,4],[8,3],[8,4],[9,3],[9,4],[10,3],[10,4],[11,4],[12,2],[12,3],[12,4],
                 [9,1]]
    numDross = 0
    for card in drossList:
        if card in CollectCard:
            numDross += 1
    if numDross >= 10:
        YakuList.append([15,'Kasu', numDross-9])
    
    # koi koi point calculate
    if KoiKoiNum > 0:
        YakuList.append([16,'Koi-Koi', KoiKoiNum])
    
    # calculate points
    RoundPts = 0
    for yaku in YakuList:
        if yaku[0] != 16:
            RoundPts += yaku[2]
    if KoiKoiNum <= 3:
        RoundPts += KoiKoiNum
    else:
        RoundPts *= KoiKoiNum - 2
    
    return (YakuList, RoundPts)


def UpdateBoardCard(boardCard, card, collectCard):
    if collectCard == []:
        for i in range(0,16):
            if boardCard[i] == [0,0]:
                boardCard[i] = card
                break
    else:
        for i in range(0,16):
            if boardCard[i] in collectCard:
                boardCard[i] = [0,0]                
    return boardCard

def UpdateBoardCardNoSlot(boardCard, card, collectCard):
    if collectCard == []:
        boardCard.append(card)
    else:
        for tmp in collectCard:
            if tmp in boardCard:
                boardCard.remove(tmp)
    return boardCard
    
    
    
    

