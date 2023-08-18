#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 01:27:02 2023

@author: dev
"""

import numpy as np
import time

def init():
    base_mat = np.zeros((3, 3), dtype='int')
    player = True      # 1st turn is of the player
    result = 0
    turn_ctr = 0
    return base_mat, player, result, turn_ctr

def action(base_mat, player):

    if player:
        while(1):
            x = np.random.randint(0, 3)
            y = np.random.randint(0, 3)
            if base_mat[x, y] == 0:
                break
        base_mat[x, y] = 1
    else:
        while(1):
            x = np.random.randint(0, 3)
            y = np.random.randint(0, 3)
            if base_mat[x, y] == 0:
                break
        base_mat[x, y] = -1

    return base_mat

def compute_result(base_mat):
    '''
    0 for game running
    1 for draw
    2 for player won
    3 for bot won
    '''
    if tuple(base_mat[0]) == (1, 1, 1) or tuple(base_mat[1]) == (1, 1, 1) or tuple(base_mat[2]) == (1, 1, 1) or tuple(base_mat.T[0]) == (1, 1, 1) or tuple(base_mat.T[1]) == (1, 1, 1) or tuple(base_mat.T[2]) == (1, 1, 1) or (base_mat[0, 0], base_mat[1, 1], base_mat[2, 2]) == (1, 1, 1) or (base_mat[0, 2], base_mat[1, 1], base_mat[2, 0]) == (1, 1, 1):
        print('Player won!')
        return 2
    elif tuple(base_mat[0]) == (-1, -1, -1) or tuple(base_mat[1]) == (-1, -1, -1) or tuple(base_mat[2]) == (-1, -1, -1) or tuple(base_mat.T[0]) == (-1, -1, -1) or tuple(base_mat.T[1]) == (-1, -1, -1) or tuple(base_mat.T[2]) == (-1, -1, -1) or (base_mat[0, 0], base_mat[1, 1], base_mat[2, 2]) == (-1, -1, -1) or (base_mat[0, 2], base_mat[1, 1], base_mat[2, 0]) == (-1, -1, -1):
        print('Bot won!')
        return 3
    elif np.min(np.min(np.absolute(base_mat))):
        print('Game Draw!')
        return 1
    else:
        return 0

def status(turn_ctr, player, base_mat):
    print('Turn count: ', turn_ctr, end=" | ")
    if player:
        print("Player's turn")
    else:
        print("Bot's turn")
    display_board(base_mat)
    return

def display_board(base_mat):
    for row in base_mat:
        for mark in row:
            if mark == 1:
                print(' X ', end="")
            elif mark == -1:
                print(' O ', end="")
            else:
                print('   ', end="")
        print('\n-----------')

def start():
    base_mat, player, result, turn_ctr = init()
    while not result:
        base_mat = action(base_mat, player)
        turn_ctr += 1
        status(turn_ctr, player, base_mat)
        player = (player != True)
        result = compute_result(base_mat)
        time.sleep(0.75)
    print('Restarting in 2 seconds. Press ctrl+c to stop.')
    time.sleep(2)
    start()

start()