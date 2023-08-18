#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 02:29:10 2023

@author: dev
"""

import gym
from gym import spaces
import numpy as np
import pickle
import os

def init_hyperparams():
    num_episodes = 100000
    lr = 0.0001
    discount_factor = 0.1
    epsilon = 0.9
    return num_episodes, lr, discount_factor, epsilon

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()

        self.action_space = spaces.Discrete(9)  # 9 possible actions (0-8)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.float32)

        self.board = np.zeros((3, 3), dtype=np.float32)
        self.player = True      # 1st turn is of the player
        self.result = 'running'  # 'running', 'draw', 'player_won', 'bot_won'
        self.turn_ctr = 0
        self.reward = 0

    def step(self, action=None):
        x, y = divmod(action, 3)
        if self.board[x, y] == 0:
            self.board[x, y] = 1 if self.player else -1
            self.result = self.compute_result()
            self.reward += self.get_reward(self.result)
            self.turn_ctr += 1
            self.player = not self.player
            done = (self.result != 'running') or (self.turn_ctr == 9)
        else:
            self.reward -= 1
            done = False

        return self.board.copy(), done, {}

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.player = True
        self.result = 'running'
        self.turn_ctr = 0
        self.reward = 0  # Reset the reward for the new episode
        return self.board.copy()


    def compute_result(self):
        if tuple(self.board[0]) == (1, 1, 1) or tuple(self.board[1]) == (1, 1, 1) or tuple(self.board[2]) == (1, 1, 1) or tuple(self.board.T[0]) == (1, 1, 1) or tuple(self.board.T[1]) == (1, 1, 1) or tuple(self.board.T[2]) == (1, 1, 1) or (self.board[0, 0], self.board[1, 1], self.board[2, 2]) == (1, 1, 1) or (self.board[0, 2], self.board[1, 1], self.board[2, 0]) == (1, 1, 1):
            return 'player_won'
        elif tuple(self.board[0]) == (-1, -1, -1) or tuple(self.board[1]) == (-1, -1, -1) or tuple(self.board[2]) == (-1, -1, -1) or tuple(self.board.T[0]) == (-1, -1, -1) or tuple(self.board.T[1]) == (-1, -1, -1) or tuple(self.board.T[2]) == (-1, -1, -1) or (self.board[0, 0], self.board[1, 1], self.board[2, 2]) == (-1, -1, -1) or (self.board[0, 2], self.board[1, 1], self.board[2, 0]) == (-1, -1, -1):
            return 'bot_won'
        elif np.min(np.min(np.absolute(self.board))):
            return 'draw'
        else:
            return 'running'

    def get_reward(self, result):
        if result == 'player_won':
            return 1
        elif result == 'bot_won':
            return -1
        elif result == 'draw':
            return -0
        else:
            return -0.01

    def status(self):
        print('Turn count: ', self.turn_ctr, end=" | ")
        if self.player:
            print("Player's turn")
        else:
            print("Bot's turn")
        self.display_board()
        return

    def display_board(self):
        for row in self.board:
            for mark in row:
                if mark == 1:
                    print(' X ', end="")
                elif mark == -1:
                    print(' O ', end="")
                else:
                    print('   ', end="")
            print('\n-----------')

    def get_valid_moves(self):
        valid_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid_moves.append(i * 3 + j)
        return valid_moves

class QLearningAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_table = {}

    def get_q_value(self, state, action):
        state_key = tuple(state.flatten())  # Convert NumPy array to a tuple
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        return self.q_table[state_key][action]

    def update_q_value(self, state, action, new_q_value):
        state_key = tuple(state.flatten())  # Convert NumPy array to a tuple
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        self.q_table[state_key][action] = new_q_value


def train_q_learning_agent(env, agent, num_episodes, lr, discount_factor, epsilon):
    wins = 0
    draws = 0
    defeats = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.random() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax([agent.get_q_value(state, a) for a in range(env.action_space.n)])

            next_state, done, _ = env.step(action)

            reward = env.reward
            total_reward += reward

            q_value = agent.get_q_value(state, action)
            next_max_q_value = np.max([agent.get_q_value(next_state, a) for a in range(env.action_space.n)])

            new_q_value = q_value + lr * (reward + discount_factor * next_max_q_value - q_value)
            agent.update_q_value(state, action, new_q_value)

            state = next_state

        if (episode+1)%5000 == 0:
            epsilon = 0.5*epsilon
            print(f"Episode {episode+1}: Total Reward = {total_reward} | Epsilon {epsilon}")
            print(f'Game stats for the last 5000 episodes:\n wins: {wins}/{5000} | {wins/50}% \n draws: {draws}/5000 | {draws/50}% \n defeats: {defeats}/5000 | {defeats/50}%')
            (wins, draws, defeats) = (0, 0, 0)
        wins += (env.compute_result() == 'player_won')
        draws += (env.compute_result() == 'draw')
        defeats += (env.compute_result() == 'bot_won')
    return agent

def play_against_bot(env, agent):
    state = env.reset()
    done = False

    while not done:
        # Your turn
        env.display_board()
        print("Your turn!")
        valid_moves = env.get_valid_moves()  # Implement this method in your environment to get a list of valid moves
        print("Valid moves:", valid_moves)
        your_action = int(input("Enter your move (0-8): "))
        state, done, _ = env.step(your_action)

        if done:
            env.display_board()
            break

        # Bot's turn
        env.display_board()
        print("Bot's turn!")
        bot_action = np.argmax([agent.get_q_value(state, a) for a in range(env.action_space.n)])
        state, done, _ = env.step(bot_action)

        if done:
            env.display_board()
            break
    print(env.compute_result())
    print("Game Over")


def start():
    env = TicTacToeEnv()
    num_actions = env.action_space.n
    if not os.path.exists('agent.pkl'):
        env = TicTacToeEnv()
        num_actions = env.action_space.n
        agent = QLearningAgent(num_actions)

        # Hyperparameters
        num_episodes, lr, discount_factor, epsilon = init_hyperparams()

        # Start training
        agent = train_q_learning_agent(env, agent, num_episodes, lr, discount_factor, epsilon)
        with open('agent.pkl', 'wb') as f:
            pickle.dump(agent, f)

    with open('agent.pkl', 'rb') as f:
        agent = pickle.load(f)
    play_against_bot(env, agent)

start()