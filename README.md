# Koi-Koi AI

Learning based AI for playing multi-round Koi-Koi hanafuda card games. Just for fun.

![Play Interface](/markdown/koikoi_play_interface.png)

## Platform

* Python
* PyTorch 1.7.1 or higher
* PySimpleGUI (for pvc playing interface)

## About Koi-Koi Hanafuda Card Games

[Hanafuda](https://en.wikipedia.org/wiki/Hanafuda) is a kind of traditional Japanese playing cards. A hanafuda deck contains 48 cards divided by 12 suits corresponding to 12 months, which are also divided into four rank-like categories with different importance. [Koi-Koi](https://en.wikipedia.org/wiki/Koi-Koi) is a kind of two-player hanafuda card game. The goal of Koi-Koi is to collect cards by matching the cards by suit, and forming specific winning hands called Yaku from the acquired pile to earn points from the opponent.

![Hanafuda Deck](/markdown/koikoi_deck.png)

## Rules & Yaku List

Koi-Koi is consisted by multiple rounds and both players start with equal points. In every round, the dealer plays first and two players discard and draw to pair and collect cards by turn until someone forms Yakus successfully. Then, he can end this round to receive points from the opponent, or claim koi-koi and continues this round to earn more yakus and points. The detailed rules and Yaku list of this project is the same as PC game [KoiKoi-Japan](https://store.steampowered.com/app/364930/KoiKoi_Japan_Hanafuda_playing_cards/) on Steam.

![Yaku List](/markdown/koikoi_yaku.png)

## Network Model & Dataset

The model is a card-state self-attention based neural network with multiple Transformer encoder layers, and is trained by supervised learning. The dataset of 200 multi-round Koi-Koi game records is also provided. 

![Model](/markdown/koikoi_net_model.png)

## Future Work

Improve the performance with reinforcement learning...

