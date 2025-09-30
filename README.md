# Snake_RL
Snake RL — Deep Q-Learning agent for Snake (PyTorch + Pygame)

Snake RL is a compact, educational project implementing a Deep Q-Network (DQN) agent that learns to play the classic Snake game. The project uses PyTorch for the neural network, Pygame for optional visualization of episodes, and Matplotlib for live plotting of training progress.

This repository is intended for people who:

want a small but complete DQN example,

want to learn how to wire an RL training loop to a custom environment,

want a starting point for RL experiments (target networks, Double DQN, prioritized replay, image-based states).

Features

Simple, interpretable 11-dimensional state representation (dangers, direction, food relative position).

Relative action space: [turn left, straight, turn right].

Replay buffer (experience replay) + batch training.

Short-memory (online) updates + long-memory (replay) updates.

Simple MLP Q-network (can run on CPU or GPU).

Pygame rendering for visualization (optional).

Live training plot with Matplotlib showing per-game score and mean score.

Model saving on new high-score.