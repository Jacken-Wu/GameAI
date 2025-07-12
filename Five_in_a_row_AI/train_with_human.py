import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import logging
import os
import pygame
from CNN_model import DQN, ReplayBuffer, GameEnv


BATCH_SIZE = 64
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000
TARGET_UPDATE = 20
SAVE_PATH = "./model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# 初始化
env = GameEnv()
policy_net = DQN().to(device)
target_net = DQN().to(device)

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

if os.path.exists(SAVE_PATH):
    checkpoint = torch.load(SAVE_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint.get('episode', 0)
    logging.info(f"Load model from {SAVE_PATH}")
else:
    start_episode = 0
logging.info(f"Start episode: {start_episode}")

target_net.load_state_dict(policy_net.state_dict())  # 同步权重
target_net.eval()

replay_buffer = ReplayBuffer(10000)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Parameters of policy_net: {count_parameters(policy_net)}")

pygame.init()
screen = pygame.display.set_mode((750, 750))


def weighted_random_action(state):
    board_size = 15

    # 找出所有空格子的位置
    valid = (state[0] + state[1] == 0).flatten()  # 形状：[225]
    valid_indices = valid.nonzero(as_tuple=True)[0]  # 空位索引，比如 [0, 1, 2, 45, 67, ...]

    # 构造每个点到中心点 (7,7) 的距离平方矩阵
    grid = torch.arange(board_size)  # [0, 1, ..., 14]
    row_coords = grid.repeat(board_size).reshape(board_size, board_size)
    col_coords = grid.repeat_interleave(board_size).reshape(board_size, board_size)
    dist2center = (row_coords - 7)**2 + (col_coords - 7)**2  # 每个格子的中心距离平方，越远值越大

    # 距离越小 → 权重越高（靠近中心）
    weights = torch.exp(-dist2center.float() / 20.0)  # 权重公式，可调参数为 20
    weights = weights.flatten()  # 展平为 1D：形状 [225]

    # 只保留合法位置的权重
    valid_weights = weights[valid]  # 提取空位对应的权重

    # 转换为概率（归一化）
    probs = valid_weights / valid_weights.sum()

    # 按照概率进行抽样，返回合法空位中的一个动作
    selected_index = torch.multinomial(probs, 1).item()

    return valid_indices[selected_index].item()


def neighbor_biased_random_action(state):
    board = state[0] + state[1]  # 所有已落子位置
    occupied = (board == 1)
    valid = (board == 0)

    # 获取所有落子位置的坐标
    occupied_indices = torch.nonzero(occupied)

    # 构造一个权重矩阵（越靠近已有子越高）
    weight = torch.zeros_like(board, dtype=torch.float32)

    for (row, col) in occupied_indices:
        # 在已落子点周围加权
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if 0 <= nr < 15 and 0 <= nc < 15 and valid[nr, nc]:
                    weight[nr, nc] += 1.0  # 每个邻近空格加权
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if 0 <= nr < 15 and 0 <= nc < 15 and valid[nr, nc]:
                    weight[nr, nc] += 0.1  # 每个邻近空格加权

    # 如果周围没有空格可落，就退回全局加权
    if weight.sum() == 0:
        return weighted_random_action(state)

    # 展平成 1D 并加 softmax 形成概率分布
    prob = weight.flatten()
    prob = prob / prob.sum()

    choice = torch.multinomial(prob, 1).item()
    return choice



def select_action(state, epsilon):
    if random.random() < epsilon:
        # valid = (state[0] + state[1] == 0).flatten()
        # valid_indices = valid.nonzero(as_tuple=True)[0]
        # return random.choice(valid_indices).item()

        # if (state[0] + state[1]).sum() == 0:  # 空棋盘，中心落子
        #     if random.random() < 0.8:
        #         return 112

        # 概率落子
        if random.random() < 0.9:
            return neighbor_biased_random_action(state)
        return weighted_random_action(state)
    
    else:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0).to(device))  # [1, 225]
            q_values = q_values.cpu().squeeze(0)
            mask = (state[0] + state[1]).flatten()
            q_values[mask == 1] = -float("inf")  # 屏蔽已下子
            return torch.argmax(q_values).item()


def draw(screen, board):
    # Draw 15x15 board
    for i in range(15):
        pygame.draw.line(screen, (0, 0, 0), (i*50+25, 25), (i*50+25, 725), 1)
        pygame.draw.line(screen, (0, 0, 0), (25, i*50+25), (725, i*50+25), 1)
    
    # Draw pieces
    for row in range(15):
        for col in range(15):
            if board[0][row][col] == 1:
                pygame.draw.circle(screen, (0, 0, 0), (col*50+25, row*50+25), 20)
            elif board[1][row][col] == 1:
                pygame.draw.circle(screen, (0, 0, 0), (col*50+25, row*50+25), 20)
                pygame.draw.circle(screen, (255, 255, 255), (col*50+25, row*50+25), 19)


# 训练主循环
for episode in range(start_episode, start_episode + 100000):
    state = env.reset()
    done = False
    total_reward = 0
    player = random.choice([0, 1])  # 随机选手
    reward_for_human = 0.0
    human_action = 112
    if player == 1:
        state, reward_for_human, _ = env.step(human_action, 1)  # 先手落子
        player = 0

    state_for_human = state.clone()
    while not done:
        # AI 落子
        state_for_model = state.clone()
        # ε-greedy 动作选择
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                torch.exp(torch.tensor(-episode / EPSILON_DECAY)).item()

        action = select_action(state_for_model, epsilon)

        # 当前 AI 落子
        next_state, reward, done = env.step(action, 0)
        player = 1
        if done:
            reward_for_human = -1.0
        
        # 人类玩家落子
        state_for_human_last = state_for_human.clone()
        state_for_human = next_state.clone()
        # 将人类的落子存入经验池
        replay_buffer.push(state_for_human_last, human_action, reward_for_human, state_for_human, done)
        while player == 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col = x // 50
                    row = y // 50
                    # 判断落子合法性
                    if state[0][row][col] == 0 and state[1][row][col] == 0:
                        human_action = row * 15 + col
                        next_state, reward_for_human, done_opponent = env.step(human_action, 1)
                        player = 0
                        if not done and done_opponent:
                            reward = -1.0
                            done = True
            screen.fill((255, 215, 125))
            draw(screen, next_state)
            pygame.display.flip()

        next_state_for_model = next_state.clone()

        # 存入经验池
        replay_buffer.push(state_for_model, action, reward, next_state_for_model, done)

        # 准备下一轮
        state = next_state_for_model
        total_reward += reward

        # 网络训练
        if len(replay_buffer) >= BATCH_SIZE:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)
            batch_state = batch_state.to(device)
            batch_next_state = batch_next_state.to(device)
            batch_action = batch_action.to(device)
            batch_reward = batch_reward.to(device)
            batch_done = batch_done.to(device)

            # Q(s, a)
            q_values = policy_net(batch_state)
            state_action_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)

            # max_a' Q_target(s', a')
            with torch.no_grad():
                next_q_values = target_net(batch_next_state)
                max_next_q_values = next_q_values.max(1)[0]

            expected_q_values = batch_reward + GAMMA * max_next_q_values * (1 - batch_done)

            loss = F.mse_loss(state_action_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 日志与模型保存
    if episode % 20 == 0:
        try:
            logging.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Loss: {loss.item():.4f}")
        except:
            logging.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Loss: no loss")
    if episode % 100 == 0:
        torch.save({
            'episode': episode,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_PATH.replace('.pth', f'_{episode}.pth'))
        logging.info(f"Model saved to {SAVE_PATH.replace('.pth', f'_{episode}.pth')}")

# 最终保存
torch.save({
    'episode': episode,
    'model_state_dict': policy_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, SAVE_PATH)
logging.info(f"Model saved to {SAVE_PATH}")
logging.info("Training finished.")
