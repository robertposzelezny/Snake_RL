import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.body[0]
        point_l = [head[0] - game.size, head[1]]
        point_r = [head[0] + game.size, head[1]]
        point_u = [head[0], head[1] - game.size]
        point_d = [head[0], head[1] + game.size]

        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        state = [
            # Danger straight
            (dir_r and game.check_collision_at(point_r)) or
            (dir_l and game.check_collision_at(point_l)) or
            (dir_u and game.check_collision_at(point_u)) or
            (dir_d and game.check_collision_at(point_d)),

            # Danger right
            (dir_u and game.check_collision_at(point_r)) or
            (dir_d and game.check_collision_at(point_l)) or
            (dir_l and game.check_collision_at(point_u)) or
            (dir_r and game.check_collision_at(point_d)),

            # Danger left
            (dir_d and game.check_collision_at(point_r)) or
            (dir_u and game.check_collision_at(point_l)) or
            (dir_r and game.check_collision_at(point_u)) or
            (dir_l and game.check_collision_at(point_d)),

            # direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food position
            game.food[0] < game.position[0],
            game.food[0] > game.position[0],
            game.food[1] < game.position[1],
            game.food[1] > game.position[1]
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
