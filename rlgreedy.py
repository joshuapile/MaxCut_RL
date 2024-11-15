import random
import numpy as np
import networkx as nx
import copy
from typing import List, Tuple
import time
from util import read_nxgraph
from util import obj_maxcut
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define parameters for Deep Q-Learning
GAMMA = 0.9          # Discount factor
EPSILON_START = 1.0  # Starting epsilon for exploration
EPSILON_END = 0.01   # Minimum epsilon
EPSILON_DECAY = 0.995  # Epsilon decay rate per episode
LR = 0.001           # Learning rate for the optimizer
BATCH_SIZE = 64      # Batch size for experience replay
MEMORY_SIZE = 10000  # Maximum size of the replay buffer
TARGET_UPDATE_FREQ = 10  # Frequency to update the target network

# Experience Replay Memory
from collections import deque
memory = deque(maxlen=MEMORY_SIZE)

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, num_nodes, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_nodes, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_nodes)  # Output Q-values for all actions

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def choose_action(state: List[int], epsilon: float, num_nodes: int, q_network: QNetwork) -> int:
    """
    Choose an action (node to flip) based on an epsilon-greedy policy.
    Returns the index of the node to flip.
    """
    if random.uniform(0, 1) < epsilon:
        # Exploration: choose a random node to flip
        return random.choice(range(num_nodes))
    else:
        # Exploitation: choose the action that maximizes Q(s,a)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: [1, num_nodes]
        q_values = q_network(state_tensor)  # Shape: [1, num_nodes]
        action = torch.argmax(q_values).item()
        return action

def optimize_model(q_network, target_network, optimizer, loss_fn):
    """
    Sample a batch from memory and perform a Q-learning update.
    """
    if len(memory) < BATCH_SIZE:
        return

    # Sample a random batch from memory
    batch = random.sample(memory, BATCH_SIZE)
    batch = list(zip(*batch))  # Transpose batch
    states, actions, rewards, next_states, dones = batch

    # Convert to tensors
    states_tensor = torch.FloatTensor(states)          # Shape: [BATCH_SIZE, num_nodes]
    actions_tensor = torch.LongTensor(actions).unsqueeze(1)  # Shape: [BATCH_SIZE, 1]
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)  # Shape: [BATCH_SIZE, 1]
    next_states_tensor = torch.FloatTensor(next_states)  # Shape: [BATCH_SIZE, num_nodes]
    dones_tensor = torch.FloatTensor(dones).unsqueeze(1)  # Shape: [BATCH_SIZE, 1]

    # Compute Q(s,a)
    q_values = q_network(states_tensor).gather(1, actions_tensor)  # Shape: [BATCH_SIZE, 1]

    # Compute target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states_tensor)
        max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
        target_q_values = rewards_tensor + GAMMA * max_next_q_values * (1 - dones_tensor)

    # Compute loss
    loss = loss_fn(q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def greedy_maxcut_dqn(init_solution: List[int], num_episodes: int, num_steps: int, graph: nx.Graph) -> Tuple[int, List[int], List[float]]:
    """
    Deep Q-Learning based Max-Cut algorithm.
    """
    print('Deep Q-Learning based Max-Cut')
    start_time = time.time()
    num_nodes = graph.number_of_nodes()

    # Initialize networks and optimizer
    q_network = QNetwork(num_nodes)
    target_network = QNetwork(num_nodes)
    target_network.load_state_dict(q_network.state_dict())  # Initialize target network
    optimizer = optim.Adam(q_network.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    epsilon = EPSILON_START
    scores = []
    max_score = None
    best_solution = None

    for episode in range(num_episodes):
        curr_solution = copy.deepcopy(init_solution)
        curr_score = obj_maxcut(curr_solution, graph)

        for step in range(num_steps):
            # Choose an action using epsilon-greedy policy
            action = choose_action(curr_solution, epsilon, num_nodes, q_network)

            # Calculate new solution by flipping the chosen node
            new_solution = copy.deepcopy(curr_solution)
            new_solution[action] = (new_solution[action] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)

            # Calculate reward as the change in score
            reward = new_score - curr_score

            # Define 'done' if you have a termination condition
            done = False  # For Max-Cut, episodes can be of fixed length

            # Store the transition in memory
            memory.append((curr_solution, action, reward, new_solution, done))

            # Move to the next state
            curr_solution = new_solution
            curr_score = new_score

            # Perform one step of the optimization
            optimize_model(q_network, target_network, optimizer, loss_fn)

            # Optionally, you can break the loop if done is True

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        scores.append(curr_score)

        # Keep track of the best solution
        if max_score is None or curr_score > max_score:
            max_score = curr_score
            best_solution = curr_solution

        print(f"Episode: {episode+1}, Score: {curr_score}, Epsilon: {epsilon}")

    running_duration = time.time() - start_time
    print('Best Score:', max_score)
    print('Best Solution (Binary):', ''.join(map(str, best_solution)))
    print('Running Duration:', running_duration)
    return max_score, best_solution, scores

if __name__ == '__main__':
    # Read data
    graph = read_nxgraph('./data/gset/gset_14.txt')

    # Parameters
    num_episodes = 50
    num_steps = 100  # Number of steps per episode

    # Initialize solution randomly
    init_solution = [random.randint(0,1) for _ in range(graph.number_of_nodes())]

    # Run DQN-based Max-Cut algorithm
    max_score, best_solution, scores = greedy_maxcut_dqn(init_solution, num_episodes, num_steps, graph)
