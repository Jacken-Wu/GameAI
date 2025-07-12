import torch
import pygame
import logging
import time
import random
from game_play import GameBoard

class GameGUI:
    def __init__(self):
        self.game_board = GameBoard()
    
    
    def draw(self, screen, win_mark, player, player_color):
        # Draw 15x15 board
        for i in range(15):
            pygame.draw.line(screen, (0, 0, 0), (i*50+25, 25), (i*50+25, 725), 1)
            pygame.draw.line(screen, (0, 0, 0), (25, i*50+25), (725, i*50+25), 1)
        
        # Draw pieces
        for row in range(15):
            for col in range(15):
                if self.game_board.board[0][row][col] == 1:
                    pygame.draw.circle(screen, player_color[0], (col*50+25, row*50+25), 20)
                elif self.game_board.board[1][row][col] == 1:
                    pygame.draw.circle(screen, (0, 0, 0), (col*50+25, row*50+25), 20)
                    pygame.draw.circle(screen, player_color[1], (col*50+25, row*50+25), 19)
                if win_mark[row][col] == 1:
                    pygame.draw.circle(screen, (255, 0, 0), (col*50+25, row*50+25), 20)
                    if player == GameBoard.PlayerAI:
                        pygame.draw.circle(screen, player_color[0], (col*50+25, row*50+25), 17)
                    else:
                        pygame.draw.circle(screen, player_color[1], (col*50+25, row*50+25), 17)
    
    
    def run(self, policy_net: torch.nn.Module, device: str):
        pygame.init()
        screen = pygame.display.set_mode((750, 750))
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 30)
        running = True
        player = random.randint(0, 1)  # Randomly choose player AI or Human first
        player_color = [(255, 255, 255), (255, 255, 255)]
        player_color[player] = (0, 0, 0)  # First player is black
        is_win = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    col = pos[0] // 50
                    row = pos[1] // 50
                    if is_win:
                        is_win = False
                        self.game_board.reset()
                    elif player == GameBoard.PlayerHuman:
                        if self.game_board.move(player, row, col):
                            player = 1 - player
                            logging.info(f"Player Human move: {row}, {col}")
                        else:
                            logging.info("Human invalid move")
            if player == GameBoard.PlayerAI:
                board = self.game_board.board.clone()
                move_array = policy_net(board.unsqueeze(0).to(device))  # [1, 2, 15, 15]
                move_array = move_array.squeeze()  # [255]
                mask = (board[0] + board[1]).flatten()
                move_array[mask == 1] = -float("inf")  # 屏蔽已下子
                action = torch.argmax(move_array).item()
                row = action // 15
                col = action % 15
                if self.game_board.move(player, row, col):
                    player = 1 - player
                    logging.info(f"Player AI move: {row}, {col}")
                else:
                    logging.info("AI invalid move")

            clock.tick(60)
            screen.fill((255, 215, 125))
            win_mark = self.game_board.check_win(GameBoard.PlayerAI)
            if win_mark.sum() == 0:
                win_mark = self.game_board.check_win(GameBoard.PlayerHuman)
                self.draw(screen, win_mark, GameBoard.PlayerHuman, player_color)
                if win_mark.sum() > 0:
                    text_surface = font.render("Player Human wins!", True, (0, 0, 0))
                    screen.blit(text_surface, (250, 300))
                    if not is_win:
                        logging.info("Player Human wins!")
                    is_win = True
            else:
                self.draw(screen, win_mark, GameBoard.PlayerAI, player_color)
                text_surface = font.render("Player AI wins!", True, (0, 0, 0))
                screen.blit(text_surface, (250, 300))
                if not is_win:
                    logging.info("Player AI wins!")
                is_win = True
            pygame.display.flip()


if __name__ == '__main__':
    game_gui = GameGUI()
    game_gui.run()