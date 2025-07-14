import logging
from game_play import GamePlay, GameGUI

game = GamePlay()
gui = GameGUI()

gui.pygame_init(game.width, game.height)

while True:
    state, score, _ = game.reset()
    gui.draw(state, score, game.direction)
    while game.game_over == False:
        action = gui.event_deal()
        if action is not None:
            game.move(action)

        # calculate the distance between the head and the food
        head_row, head_col = game.get_head_position()
        food_indices = game.state[GamePlay.FOOD].nonzero()
        distance_reward_last = 0.0
        for food_pos in food_indices:
            food_row, food_col = food_pos[0], food_pos[1]
            distance = abs(head_row - food_row) + abs(head_col - food_col)
            distance_reward_last += ((40 - distance) / 40)
            
        state, score, game_over = game.step()
        gui.draw(state, score, game.direction)
        
        head_row, head_col = game.get_head_position()
        distance_reward = 0.0
        for food_pos in food_indices:
            food_row, food_col = food_pos[0], food_pos[1]
            distance = abs(head_row - food_row) + abs(head_col - food_col)
            distance_reward += ((40 - distance) / 40)
        reward = (distance_reward - distance_reward_last) * 2.5
        logging.debug("Step reward: {}".format(reward))
        
        gui.clock.tick(3)
    break