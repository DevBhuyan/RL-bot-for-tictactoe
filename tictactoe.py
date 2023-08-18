#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 01:27:02 2023

@author: dev
"""

import numpy as np
import pickle
import os
from tqdm import tqdm
from copy import deepcopy

class Board():
    def __init__(self):
        self.base_mat = np.zeros((3, 3), dtype='int')
        self.player = True      # 1st turn is of the player
        self.result = 0
        self.turn_ctr = 0

    def reset(self):
        self.base_mat = np.zeros((3, 3), dtype='int')
        self.player = True
        self.result = 0
        self.turn_ctr = 0

    def step(self, action=None):
        if not isinstance(action, int):
            action = np.random.choice(self.valid_moves())
        x, y = divmod(action, 3)
        self.base_mat[x, y] = 1 if self.player else -1
        self.turn_ctr += 1
        # self.status()
        self.x = x
        self.y = y
        self.compute_result()
        self.player = not self.player

    def compute_result(self):
        '''
        Bot wins get rewards
        '''
        x = self.x
        y = self.y
        base_mat = self.base_mat
        if (tuple(base_mat[x]) == (1, 1, 1) or
            tuple(base_mat.T[y]) == (1, 1, 1) or
            (base_mat[0, 0], base_mat[1, 1], base_mat[2, 2]) == (1, 1, 1) or
            (base_mat[0, 2], base_mat[1, 1], base_mat[2, 0]) == (1, 1, 1)):
            print('Player won!')
            self.result = -5/self.turn_ctr
        elif (tuple(base_mat[x]) == (-1, -1, -1) or
              tuple(base_mat.T[y]) == (-1, -1, -1) or
              (base_mat[0, 0], base_mat[1, 1], base_mat[2, 2]) == (-1, -1, -1) or
              (base_mat[0, 2], base_mat[1, 1], base_mat[2, 0]) == (-1, -1, -1)):
            print('Bot won!')
            self.result = 1/self.turn_ctr
        elif self.turn_ctr == 9:
            print('Game Draw!')
            self.result = -0.05

    def status(self):
        print('Turn count: ', self.turn_ctr, end=" | ")
        print("Player's turn") if self.player else print("Bot's turn")
        self.display_board()
        return

    def display_board(self):
        for row in self.base_mat:
            for mark in row:
                if mark == 1:
                    print(' X ', end="")
                elif mark == -1:
                    print(' O ', end="")
                else:
                    print('   ', end="")
            print('\n-----------')

    def valid_moves(self):
        valid_moves = []
        for i in range(3):
            for j in range(3):
                if self.base_mat[i, j] == 0:
                    valid_moves.append(i * 3 + j)
        return valid_moves

class Agent():
    # 9! = 362880, we need atmost these many entries in the state_paths
    # TODO: Find a mapping between current state and next state and compute average reward per step
    '''
    First iterate through the total number of paths and compute reward.
    Distribute reward to each step of path. Now each step in a path must have a distributed reward score
    The remaining step is to map the step-wise changes to a reward system
    '''
    def __init__(self):
        self.step_scores = []
    def convert_state_path_on_the_go(self, path):
        steps, reward = path[0], path[1]
        for i in range(len(steps)//2):
            move = [tuple(steps[2*i].flatten()), tuple(steps[2*i+1].flatten()), reward]
            if move[:-1] not in [m[:-1] for m in self.step_scores]:
                self.step_scores.append(move)   # Structure of move is [current, next, reward]
            else:
                pass
                # ADD SOME CODE TO AVERAGE REWARDS IF SAME MOVE GETS DIFFERENT REWARDS
                idx = [m[:-1] for m in self.step_scores].index(move[:-1])
                self.step_scores.append([move[0], move[1], (move[2]+self.step_scores[idx][2])/2])
                # FIXME: CHECK IF THIS CODE WORKS
    def recommend_best_action(self, base_mat, board):
        max_reward = 0
        best_config = None
        flag = 1
        for move in self.step_scores:
            if move[0] == tuple(base_mat.flatten()):
                flag = 0
                # IF REWARD IS POSITIVE. I.E. BOT REMEMBERS A WINNING MOVE
                if move[2] > max_reward:
                    max_reward = move[2]
                    best_config = move[1]
        if best_config == None:
            # BOT DOESN'T KNOW ANY WINNING MOVE
            # FIND THE MOVES WHERE BOT WON'T WIN, DO SOMETHING OTHER THAN THOSE
            dont_take_this_no = []
            for move in self.step_scores:
                if move[0] == tuple(base_mat.flatten()):
                    flag = 0
                    if move[2] < 0:
                        # DON'T DO THIS MOVE
                        dont_take_this_no.append(np.argmax(np.array(list(move[1])) - np.array(base_mat.flatten())))
            best_action = np.random.choice([i for i in board.valid_moves() if i not in dont_take_this_no])
        else:
            best_action = np.argmax(np.array(list(best_config)) - np.array(base_mat.flatten()))
        if flag:
            return None
        else:
            return best_action

def learn_from_player(agent, board):
    board.reset()
    state_path = []
    while board.result == 0:
        board.display_board()
        print("Your turn!")
        valid_moves = board.valid_moves()
        print("Valid moves:", valid_moves)
        your_action = int(input("Enter your move (0-8): "))
        board.step(your_action)

        state_path.append(deepcopy(board.base_mat))

        if board.result != 0:
            board.display_board()
            break

        # Bot's turn
        board.display_board()
        print("Bot's turn!")

        if agent.recommend_best_action(deepcopy(board.base_mat), board) == None:
            bot_action = np.random.choice(board.valid_moves())
        else:
            bot_action = agent.recommend_best_action(deepcopy(board.base_mat), board)

        board.step(bot_action)

        state_path.append(deepcopy(board.base_mat))

        if board.result != 0:
            board.display_board()
            break

    board.compute_result()
    print("Game Over")
    agent.convert_state_path_on_the_go([state_path, board.result])
    return agent

def start():
    print('You: (X)')
    print('Bot: (O)')
    board = Board()
    agent = Agent()

    if os.path.exists('./agent.pkl'):
        agent = pickle.load(open('agent.pkl', 'rb'))

    for _ in range(10):
        agent = learn_from_player(agent, board)

    pickle.dump(agent, open('agent.pkl', 'wb'))

start()