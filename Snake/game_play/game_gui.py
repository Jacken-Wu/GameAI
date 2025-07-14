import pygame
from game_play.game_play import GamePlay

class GameGUI:
    BLOCK_SIZE = 40
    BACKGROUND_COLOR = (255, 255, 255)
    LINE_COLOR = (0, 0, 0)
    HEAD_COLOR = (0, 0, 255)
    BODY_COLOR = (0, 255, 0)
    FOOD_COLOR = (255, 0, 0)
    TEXT_COLOR = (0, 0, 0)
    
    def __init__(self):
        pass
    
    def pygame_init(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width*GameGUI.BLOCK_SIZE, height*GameGUI.BLOCK_SIZE+20))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
    
    def draw(self, state, score, direction):
        """
        @param state: state of the game, shape: (4, height, width)
        """
        height = state.shape[1]
        width = state.shape[2]
        self.screen.fill(GameGUI.BACKGROUND_COLOR)
        for i in range(height):
            for j in range(width):
                # Draw blocks
                pygame.draw.lines(self.screen, GameGUI.LINE_COLOR, False, [(j*GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE), (j*GameGUI.BLOCK_SIZE+GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE), (j*GameGUI.BLOCK_SIZE+GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE+GameGUI.BLOCK_SIZE), (j*GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE+GameGUI.BLOCK_SIZE)], 1)
                # Draw snake and food
                if state[GamePlay.BODY][i][j] == 1:
                    pygame.draw.rect(self.screen, GameGUI.BODY_COLOR, (j*GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE, GameGUI.BLOCK_SIZE, GameGUI.BLOCK_SIZE))
                if state[GamePlay.FOOD][i][j] == 1:
                    pygame.draw.rect(self.screen, GameGUI.FOOD_COLOR, (j*GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE, GameGUI.BLOCK_SIZE, GameGUI.BLOCK_SIZE))
                if state[GamePlay.HEAD][i][j] == 1:
                    pygame.draw.rect(self.screen, GameGUI.HEAD_COLOR, (j*GameGUI.BLOCK_SIZE, i*GameGUI.BLOCK_SIZE, GameGUI.BLOCK_SIZE, GameGUI.BLOCK_SIZE))
        # Draw score
        score_text = self.font.render("Score: "+str(score), True, GameGUI.TEXT_COLOR)
        self.screen.blit(score_text, (10, height*GameGUI.BLOCK_SIZE))
        # Draw direction
        if direction == GamePlay.UP:
            direction_text = self.font.render("Direction: UP", True, GameGUI.TEXT_COLOR)
        elif direction == GamePlay.DOWN:
            direction_text = self.font.render("Direction: DOWN", True, GameGUI.TEXT_COLOR)
        elif direction == GamePlay.LEFT:
            direction_text = self.font.render("Direction: LEFT", True, GameGUI.TEXT_COLOR)
        elif direction == GamePlay.RIGHT:
            direction_text = self.font.render("Direction: RIGHT", True, GameGUI.TEXT_COLOR)
        self.screen.blit(direction_text, (200, height*GameGUI.BLOCK_SIZE))
        # Update screen
        pygame.display.update()
    
    def event_deal(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return GamePlay.UP
                elif event.key == pygame.K_DOWN:
                    return GamePlay.DOWN
                elif event.key == pygame.K_LEFT:
                    return GamePlay.LEFT
                elif event.key == pygame.K_RIGHT:
                    return GamePlay.RIGHT
        return None