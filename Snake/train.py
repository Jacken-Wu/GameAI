import torch
import torch.nn.functional as F
import os
import logging
from game_play import GameEnv, SnakeNet, ReplayBuffer, Config, GameGUI

BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000
TARGET_UPDATE = 10

model_path = Config.model_path
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_path += Config.model_name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

env = GameEnv()
gui = GameGUI()
policy_net = SnakeNet().to(device)
target_net = SnakeNet().to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-5)
buffer = ReplayBuffer(BUFFER_SIZE)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode']
    logging.info(f"Load model from {model_path}")
else:
    start_episode = 0
logging.info(f"Start episode: {start_episode}")

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Parameters of policy_net: {count_parameters(policy_net)}")


def select_action(state, epsilon):
    if torch.rand(1).item() < epsilon:
        return torch.randint(4, 8, (1,)).item()
    else:
        with torch.no_grad():
            return policy_net(state.unsqueeze(0)).max(1)[1].item()+4


gui.pygame_init(env.game_play.width, env.game_play.height)

for episode in range(start_episode, 100000):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                torch.exp(torch.tensor(-episode / EPSILON_DECAY)).item()
        action = select_action(state.to(device), epsilon)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        gui.event_deal()
        gui.draw(state, env.score, action)

    if len(buffer) > BATCH_SIZE:
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = buffer.sample(BATCH_SIZE)
        batch_state = batch_state.to(device)
        batch_action = batch_action.to(device)
        batch_reward = batch_reward.to(device)
        batch_next_state = batch_next_state.to(device)
        batch_done = batch_done.to(device)
        
        q_values = policy_net(batch_state)
        state_action_values = q_values.gather(1, batch_action.unsqueeze(1)-4).squeeze(1)
        
        with torch.no_grad():
            next_q_values = target_net(batch_next_state)
            next_state_values = next_q_values.max(1)[0]
        
        expected_q_values = batch_reward + (GAMMA * next_state_values * (1 - batch_done))

        loss = F.mse_loss(state_action_values, expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        optimizer.step()
    
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if episode % 100 == 0:
        try:
            logging.info(f"Episode {episode}, Total reward: {total_reward:.3f}, Epsilon: {epsilon:.3f}, Loss: {loss:.3f}")
        except:
            logging.info(f"Episode {episode}, Total reward: {total_reward:.3f}, Epsilon: {epsilon:.3f}")

    if episode % 1000 == 0:
        torch.save({
            'episode': episode,
           'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path.replace('.pth', f'_{episode}.pth'))
        logging.info(f"Save model to {model_path.replace('.pth', f'_{episode}.pth')}")
        
        torch.save({
            'episode': episode,
           'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)