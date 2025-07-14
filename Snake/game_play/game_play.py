import torch
import random
# import logging

class GamePlay:
    HEAD = 0
    BODY = 1
    FOOD = 2
    ORDER = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7
    
    def __init__(self, width=25, height=15):
        self.width = width
        self.height = height
        self.state = torch.zeros(4, height, width)  # channel 0 for snake's head, channel 1 for body, channel 2 for food, channel 3 for recording the order of body
        self.state[GamePlay.HEAD, height//2, width//2] = 1  # snake's head
        self.state[GamePlay.BODY, height//2, width//2-1] = 1  # snake's body
        self.state[GamePlay.ORDER, height//2, width//2-1] = 1  # snake's body order
        self.state[GamePlay.BODY, height//2, width//2-2] = 1  # snake's body
        self.state[GamePlay.ORDER, height//2, width//2-2] = 2  # snake's body order
        while self.state[GamePlay.FOOD].sum() < 3:  # spawn 3 foods
            self._spawn_food()
        self.direction = GamePlay.RIGHT
        self.score = 0
        self.game_over = False

    def reset(self):
        width = self.width
        height = self.height
        self.state = torch.zeros(4, height, width)  # channel 0 for snake's head, channel 1 for body, channel 2 for food, channel 3 for recording the order of body
        self.state[GamePlay.HEAD, height//2, width//2] = 1  # snake's head
        self.state[GamePlay.BODY, height//2, width//2-1] = 1  # snake's body
        self.state[GamePlay.ORDER, height//2, width//2-1] = 1  # snake's body order
        self.state[GamePlay.BODY, height//2, width//2-2] = 1  # snake's body
        self.state[GamePlay.ORDER, height//2, width//2-2] = 2  # snake's body order
        while self.state[GamePlay.FOOD].sum() < 3:  # spawn 3 foods
            self._spawn_food()
        self.direction = GamePlay.RIGHT
        self.score = 0
        self.game_over = False
        return self.state, self.score, self.game_over
        
    def _spawn_food(self):
        while True:
            food_row = random.randint(0, self.height-1)
            food_col = random.randint(0, self.width-1)
            if self.state[GamePlay.HEAD, food_row, food_col] == 0 and self.state[GamePlay.BODY, food_row, food_col] == 0 and self.state[GamePlay.FOOD, food_row, food_col] == 0:
                self.state[GamePlay.FOOD, food_row, food_col] = 1
                break
    
    def move(self, direction):
        """
        @return: True if the move is valid, False otherwise
        """
        if self.direction == GamePlay.UP and direction == GamePlay.DOWN:
            return False
        elif self.direction == GamePlay.DOWN and direction == GamePlay.UP:
            return False
        elif self.direction == GamePlay.LEFT and direction == GamePlay.RIGHT:
            return False
        elif self.direction == GamePlay.RIGHT and direction == GamePlay.LEFT:
            return False
        self.direction = direction
        return True
    
    def get_head_position(self):
        """
        @return: (row, col) of the snake's head
        """
        indices = torch.where(self.state[GamePlay.HEAD] == 1)
        return indices[0][0], indices[1][0]
    
    def _get_tail_position(self):
        """
        @return: (row, col) of the end of the snake's body
        """
        indices = torch.argmax(self.state[GamePlay.ORDER])
        return indices // self.width, indices % self.width
    
    def step(self):
        """
        @return: (state, score, game_over)
        """
        if self.game_over:
            return self.state, self.score, self.game_over

        # move snake
        ## compute next head position
        head_row, head_col = self.get_head_position()
        # logging.debug("Head position: ({}, {})".format(head_row, head_col))
        
        if self.direction == GamePlay.UP:
            next_head_row = head_row - 1
            next_head_col = head_col
            if head_row == 0:
                self.game_over = True

        elif self.direction == GamePlay.DOWN:
            next_head_row = head_row + 1
            next_head_col = head_col
            if head_row == self.height-1:
                self.game_over = True

        elif self.direction == GamePlay.LEFT:
            next_head_row = head_row
            next_head_col = head_col - 1
            if head_col == 0:
                self.game_over = True

        elif self.direction == GamePlay.RIGHT:
            next_head_row = head_row
            next_head_col = head_col + 1
            if head_col == self.width-1:
                self.game_over = True

        ## move snake
        if self.game_over:  # out of index (touch the wall)
            # logging.debug("Game over: snake hit the wall")
            return self.state, self.score, self.game_over
        else:
            # logging.debug("Move head to ({}, {})".format(next_head_row, next_head_col))
            # move head
            self.state[GamePlay.HEAD, head_row, head_col] = 0
            self.state[GamePlay.HEAD, next_head_row, next_head_col] = 1
            # move order
            order_mask = self.state[GamePlay.ORDER] > 0
            self.state[GamePlay.ORDER][order_mask] += 1
            self.state[GamePlay.ORDER, head_row, head_col] = 1
            # move body
            self.state[GamePlay.BODY, head_row, head_col] = 1
        
        ## check if snake eats food
        if self.state[GamePlay.FOOD, next_head_row, next_head_col] == 1:
            # logging.debug("Snake eats food.")
            self.state[GamePlay.FOOD, next_head_row, next_head_col] = 0
            self.score += 1
            self._spawn_food()
        else:
            # move tail
            tail_row, tail_col = self._get_tail_position()
            # logging.debug(f"Move tail: {(tail_row, tail_col)}")
            self.state[GamePlay.BODY, tail_row, tail_col] = 0
            self.state[GamePlay.ORDER, tail_row, tail_col] = 0
        
        ## check if snake collides with itself
        if self.state[GamePlay.BODY, next_head_row, next_head_col] == 1:
            # logging.debug("Game over: snake collides with itself")
            self.game_over = True
        
        # logging.debug("Step done.")
        return self.state, self.score, self.game_over
