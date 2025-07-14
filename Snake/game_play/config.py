import json

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

class Config:
    def __init__(self):
        pass

for key, value in config.items():
    setattr(Config, key, value)