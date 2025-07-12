import os
import torch
import torch.optim as optim
from CNN_model import DQN
from game_play import GameGUI

LOAD_PATH = "./model/model_2100.pth"

policy_net = DQN()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(LOAD_PATH):
    checkpoint = torch.load(LOAD_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint.get('episode', 0)
    print(f"Load model from {LOAD_PATH}")
else:
    start_episode = 0

policy_net.to(device)
policy_net.eval()


game_gui = GameGUI()
game_gui.run(policy_net, device)
