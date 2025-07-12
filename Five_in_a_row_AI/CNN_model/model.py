import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from math import exp
from game_play.game_play import GameBoard


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d((5, 5))  # 将特征图压缩成 5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 225)

    def forward(self, x):  # x shape: (batch_size, 2, 15, 15)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # 输出 shape: (batch_size, 225)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.stack(state),
                torch.tensor(action),
                torch.tensor(reward, dtype=torch.float32),
                torch.stack(next_state),
                torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


class GameEnv:
    def __init__(self):
        self.board = GameBoard()

    def reset(self):
        self.board.reset()
        return self.board.board.clone()  # [2,15,15]

    def step(self, action, player):
        row, col = action // 15, action % 15
        if self.board.board[0][row][col] == 1 or self.board.board[1][row][col] == 1:
            return self.board.board.clone(), -1.0, True  # 非法动作立即惩罚
        
        self.board.move(player, row, col)

        # 检查是否获胜
        win_marker = self.board.check_win(player)
        done = torch.any(win_marker).item()
        reward = 2.0 if done else 0.0

        # 检查是否死局
        if win_marker.sum() == 225:
            return self.board.board.clone(), 0.0, True  # 死局惩罚

        # 尽量下在棋盘中间
        if self.board.board.sum() < 10:
            dist = (row - 7.0) ** 2 + (col - 7.0) ** 2
            reward += exp(-dist / 20.0) * 0.2 - 0.05  # 越靠中心奖励越高

        # 连珠奖励
        if self.board.check_4_in_a_row(player, row, col):
            reward += 0.15
        elif self.board.check_3_in_a_row(player, row, col):
            reward += 0.1
        
        # 堵对手奖励
        if self.board.block_opponent_4(player, row, col):
            reward += 0.2
        elif self.board.block_opponent_3(player, row, col):
            reward += 0.05
            
        # 不相邻惩罚
        if self.board.board[player][max(0, row - 1):min(15, row + 2), max(0, col - 1):min(15, col + 2)].sum() < 2 and self.board.board.sum() > 1:
            reward -= 0.1
        elif self.board.board[player][max(0, row - 1):min(15, row + 2), max(0, col - 1):min(15, col + 2)].sum() >= 2:
            reward += 0.1
            if self.board.board.sum() == 2:
                reward += 0.5
        
        # 边界惩罚
        border_penalty = max(0, (row - 7.0) ** 2 + (col - 7.0) ** 2 - 25)
        reward -= 0.003 * border_penalty

        # reward = torch.tanh(torch.tensor(reward)).item() * 2
        return self.board.board.clone(), reward, done
