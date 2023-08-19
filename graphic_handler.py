#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 01:45:27 2023

@author: dev
"""

import pygame
from tictactoe import *

pygame.init()
screen_width = 400
screen_height = 400
board_width, board_height = screen_width, screen_height
cell_width = board_width / 3
cell_height = board_height / 3

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tic-Tac-Toe")

agent = load_dump_agent()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            column_index = int(mouse_x // cell_width)
            row_index = int(mouse_y // cell_height)
            player_action = row_index * 3 + column_index

            # Update game logic and display

            # Check for game over and display result
            # Save agent's state if needed

    # Update display

pygame.quit()