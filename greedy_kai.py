import random
import numpy as np
import networkx as nx
import copy
from typing import List, Union
import time
from util import read_nxgraph
from util import obj_maxcut
from util import transfer_nxgraph_to_weightmatrix


# Define parameters for Q-learning
ALPHA = 0.2  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate

# Q-table to store Q-values for state-action pairs
Q_table = {}

def state_to_tuple(state: List[int]) -> tuple:
    return tuple(state)

def choose_action(state: List[int], graph: nx.Graph) -> int:
    """
    Choose an action based on an epsilon-greedy policy.
    """
    if random.uniform(0, 1) < EPSILON:
        # Exploration: choose a random action
        return random.choice(range(len(state)))
    else:
        # Exploitation: choose the best action based on Q-values
        state_tuple = state_to_tuple(state)
        if state_tuple not in Q_table:
            Q_table[state_tuple] = np.zeros(len(state))
        return int(np.argmax(Q_table[state_tuple]))

def update_q_value(state: List[int], action: int, reward: float, next_state: List[int]):
    """
    Update Q-value using the Q-learning update rule.
    """
    state_tuple = state_to_tuple(state)
    next_state_tuple = state_to_tuple(next_state)

    if state_tuple not in Q_table:
        Q_table[state_tuple] = np.zeros(len(state))
    if next_state_tuple not in Q_table:
        Q_table[next_state_tuple] = np.zeros(len(next_state))

    best_next_action = np.argmax(Q_table[next_state_tuple])
    td_target = reward + GAMMA * Q_table[next_state_tuple][best_next_action]
    td_error = td_target - Q_table[state_tuple][action]
    Q_table[state_tuple][action] += ALPHA * td_error

def greedy_maxcut_rl(init_solution: List[int], num_steps: int, graph: nx.Graph) -> (int, List[int], List[int]):
    print('RL-based Greedy Max-Cut')
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []

    for iteration in range(num_steps):
        # Choose an action (node to flip) using epsilon-greedy policy
        action = choose_action(curr_solution, graph)

        # Calculate new solution by flipping the chosen node
        new_solution = copy.deepcopy(curr_solution)
        new_solution[action] = (new_solution[action] + 1) % 2
        new_score = obj_maxcut(new_solution, graph)

        # Calculate reward as the change in score
        reward = new_score - curr_score

        # Update Q-value
        update_q_value(curr_solution, action, reward, new_solution)

        # Update current solution if the new score is better
        if new_score > curr_score:
            curr_score = new_score
            curr_solution = new_solution
            scores.append(curr_score)
        else:
            # Even if the new solution is not better, we move to it sometimes to encourage exploration
            if random.uniform(0, 1) < EPSILON:
                curr_score = new_score
                curr_solution = new_solution

        print(f"Iteration: {iteration}, Score: {curr_score}")

    running_duration = time.time() - start_time
    print('Final Score:', curr_score)
    print('Final Solution (Binary):', ''.join(map(str, curr_solution)))
    print('Running Duration:', running_duration)
    return curr_score, curr_solution, scores

# Example usage
if __name__ == '__main__':
    # read data
    # graph1 = read_as_networkx_graph('data/gset_14.txt')
    graph = read_nxgraph('./data/syn/syn_50_176.txt')


    # run alg
    # init_solution = [1, 0, 1, 0, 1]
    init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))
    rw_score, rw_solution, rw_scores = random_walk(init_solution=init_solution, num_steps=1000, max_num_flips=20, graph=graph)