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
LR = 0.01            # Adjusted learning rate for the optimizer
BATCH_SIZE = 64      # Batch size for experience replay
MEMORY_SIZE = 10000  # Maximum size of the replay buffer
TARGET_UPDATE_FREQ = 10  # Frequency to update the target network

# Experience Replay Memory
from collections import deque
memory = deque(maxlen=MEMORY_SIZE)


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Output Q-values for two actions

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # Output shape: [batch_size, 2]


def get_node_features(node, solution, graph):
    neighbors = list(graph.neighbors(node))
    current_partition = solution[node]
    opposite_partition = 1 - current_partition
    intra_weight = sum(float(graph[node][nbr].get('weight', 1.0)) for nbr in neighbors if solution[nbr] == current_partition)
    inter_weight = sum(float(graph[node][nbr].get('weight', 1.0)) for nbr in neighbors if solution[nbr] == opposite_partition)
    degree = graph.degree[node]
    # Potential gain in cut value if node is flipped
    potential_gain = inter_weight - intra_weight
    return [degree, intra_weight, inter_weight, potential_gain]



def choose_action(node_features, epsilon, q_network):
    if random.uniform(0, 1) < epsilon:
        # Exploration: randomly decide to flip or not flip
        return random.choice([0, 1])  # 0: Do not flip, 1: Flip
    else:
        # Exploitation: choose action with highest Q-value
        features_tensor = torch.FloatTensor(node_features).unsqueeze(0)  # Shape: [1, input_size]
        q_values = q_network(features_tensor)  # Shape: [1, 2]
        action = torch.argmax(q_values, dim=1).item()
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
    states_tensor = torch.FloatTensor(states)          # Shape: [BATCH_SIZE, input_size]
    actions_tensor = torch.LongTensor(actions).unsqueeze(1)  # Shape: [BATCH_SIZE, 1]
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)  # Shape: [BATCH_SIZE, 1]
    next_states_tensor = torch.FloatTensor(next_states)  # Shape: [BATCH_SIZE, input_size]
    dones_tensor = torch.FloatTensor(dones).unsqueeze(1)  # Shape: [BATCH_SIZE, 1]

    # Compute Q(s,a)
    q_values_all = q_network(states_tensor)  # Shape: [BATCH_SIZE, 2]
    q_values = q_values_all.gather(1, actions_tensor)  # Shape: [BATCH_SIZE, 1]

    # Compute target Q-values
    with torch.no_grad():
        next_q_values_all = target_network(next_states_tensor)  # Shape: [BATCH_SIZE, 2]
        max_next_q_values, _ = torch.max(next_q_values_all, dim=1, keepdim=True)  # Shape: [BATCH_SIZE, 1]
        target_q_values = rewards_tensor + GAMMA * max_next_q_values * (1 - dones_tensor)

    # Compute loss
    loss = loss_fn(q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def greedy_maxcut_dqn(init_solution: List[int], num_episodes: int, graph: nx.Graph) -> Tuple[int, List[int], List[float]]:
    """
    Deep Q-Learning based Max-Cut algorithm.
    """
    print('Deep Q-Learning based Max-Cut')
    start_time = time.time()
    num_nodes = graph.number_of_nodes()

    # Initialize networks and optimizer
    input_size = 4  # Number of node features
    q_network = QNetwork(input_size)
    target_network = QNetwork(input_size)
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

        for node in range(num_nodes):
            node_features = get_node_features(node, curr_solution, graph)
            action = choose_action(node_features, epsilon, q_network)

            # Apply action
            if action == 1:  # Flip node
                new_solution = copy.deepcopy(curr_solution)
                new_solution[node] = 1 - new_solution[node]  # Flip node
                new_score = obj_maxcut(new_solution, graph)
                reward = new_score - curr_score
                done = False  # No termination condition

                # Get next node features
                next_node_features = get_node_features(node, new_solution, graph)

                # Store the transition
                memory.append((node_features, action, reward, next_node_features, done))

                # Update current solution and score
                curr_solution = new_solution
                curr_score = new_score
            else:  # Do not flip
                reward = 0
                done = False
                next_node_features = node_features  # No change in state

                # Store the transition
                memory.append((node_features, action, reward, next_node_features, done))

            # Perform optimization
            optimize_model(q_network, target_network, optimizer, loss_fn)

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
    graph = read_nxgraph('./data/syn/syn_50.txt')

    # Parameters
    num_episodes = 50

    # Initialize solution randomly
    init_solution = [random.randint(0, 1) for _ in range(graph.number_of_nodes())]

    # Run DQN-based Max-Cut algorithm
    max_score, best_solution, scores = greedy_maxcut_dqn(init_solution, num_episodes, graph)
