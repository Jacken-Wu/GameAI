import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from game_play.game_play import GamePlay


class SnakeNet(nn.Module):
    def __init__(self, height=15, width=25):  # [batch_size, 4, height, width]
        super(SnakeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)  # [batch_size, 64, height, width]
        # self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((3, 5))
        self.fc1 = nn.Linear(64*3*5, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.dp = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 4)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.capacity.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.capacity, batch_size))
        return torch.stack(state), torch.tensor(action), torch.tensor(reward, dtype=torch.float32), torch.stack(next_state), torch.tensor(done, dtype=torch.float32)
    
    def __len__(self):
        return len(self.capacity)
    

class GameEnv:
    def __init__(self, width=25, height=15):
        self.game_play = GamePlay(width, height)
        self.score = 0
    
    def reset(self):
        self.game_play.reset()
        self.score = 0
        return self.game_play.state
    
    def step(self, action):
        if self.game_play.move(action) == False:
            return self.game_play.state, -1.0, True  # invalid action

        # calculate the distance between the head and the food before moving
        head_row, head_col = self.game_play.get_head_position()
        food_indices = self.game_play.state[GamePlay.FOOD].nonzero()
        distances = []
        distance_0 = abs(head_row - food_indices[0][0]) + abs(head_col - food_indices[0][1])
        distance_1 = abs(head_row - food_indices[1][0]) + abs(head_col - food_indices[1][1])
        distance_2 = abs(head_row - food_indices[2][0]) + abs(head_col - food_indices[2][1])
        distances.append(distance_0)
        distances.append(distance_1)
        distances.append(distance_2)
        # choose the closest food
        selected_food_index = random.choice([0, 1, 2])
        if distance_0 < distance_1 and distance_0 < distance_2:
            selected_food_index = 0
            food_row, food_col = food_indices[0][0], food_indices[0][1]
        elif distance_1 < distance_0 and distance_1 < distance_2:
            selected_food_index = 1
            food_row, food_col = food_indices[1][0], food_indices[1][1]
        else:
            selected_food_index = 2
            food_row, food_col = food_indices[2][0], food_indices[2][1]

        state, score, game_over = self.game_play.step()
        if game_over:
            return state, -3.0, game_over
        reward = (score - self.score) * 10.0
        self.score = score

        # calculate the distance between the head and the food after moving
        head_row, head_col = self.game_play.get_head_position()
        distance = abs(head_row - food_row) + abs(head_col - food_col)
        if distance < distances[selected_food_index]:
            reward += 0.3
        elif distance > distances[selected_food_index]:
            reward -= 0.2
        reward -= 0.01

        return state, reward, game_over
        
