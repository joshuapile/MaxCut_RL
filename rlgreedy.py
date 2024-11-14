import random
import numpy as np
import networkx as nx
import copy
from typing import List, Dict, Tuple
import time
from collections import defaultdict
from util import read_nxgraph
from util import obj_maxcut
from util import transfer_nxgraph_to_weightmatrix

# Define parameters for Q-learning
ALPHA = 0.2  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate

# Q-table to store Q-values for state-action pairs
Q_table: Dict[Tuple[Tuple[int, ...], int], float] = defaultdict(float)

def state_to_tuple(state: List[int]) -> Tuple[int, ...]:
    """
    Converts a list state to a tuple for use as a dictionary key.
    """
    return tuple(state)

def choose_action(state: List[int]) -> int:
    """
    Choose an action (node to flip) based on an epsilon-greedy policy.
    Returns the index of the node to flip.
    """
    state_tuple = state_to_tuple(state)
    num_nodes = len(state)
    if random.uniform(0, 1) < EPSILON:
        # Exploration: choose a random node to flip
        return random.choice(range(num_nodes))
    else:
        # Exploitation: choose the action that maximizes Q(s,a)
        q_values = []
        for action in range(num_nodes):
            q_value = Q_table[(state_tuple, action)]
            q_values.append(q_value)
        # Choose the action with the highest Q-value
        return int(np.argmax(q_values))

def update_q_value(state: List[int], action: int, reward: float, next_state: List[int]):
    """
    Update Q-value using the Q-learning update rule.
    """
    state_tuple = state_to_tuple(state)
    next_state_tuple = state_to_tuple(next_state)
    
    # Get the current Q-value
    current_q = Q_table[(state_tuple, action)]
    # Find the maximum Q-value for the next state over all possible actions
    num_nodes = len(state)
    max_next_q = max([Q_table[(next_state_tuple, a)] for a in range(num_nodes)])
    # Update the Q-value
    Q_table[(state_tuple, action)] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)

def greedy_maxcut_rl(init_solution: List[int], num_steps: int, graph: nx.Graph) -> Tuple[int, List[int], List[int]]:
    """
    RL-based Greedy Max-Cut algorithm.
    """
    print('RL-based Greedy Max-Cut')
    start_time = time.time()
    num_nodes = graph.number_of_nodes()
    
    curr_solution = copy.deepcopy(init_solution)
    curr_score = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = [curr_score]
    
    for iteration in range(num_steps):
        # Choose an action (node to flip) using epsilon-greedy policy
        action = choose_action(curr_solution)
        
        # Calculate new solution by flipping the chosen node
        new_solution = copy.deepcopy(curr_solution)
        new_solution[action] = (new_solution[action] + 1) % 2
        new_score = obj_maxcut(new_solution, graph)
        
        # Calculate reward as the change in score
        reward = new_score - curr_score
        
        # Update Q-value
        update_q_value(curr_solution, action, reward, new_solution)
        
        # Update current solution
        curr_solution = new_solution
        curr_score = new_score
        scores.append(curr_score)
        
        print(f"Iteration: {iteration+1}, Score: {curr_score}")
    
    running_duration = time.time() - start_time
    print('Initial Score:', init_score)
    print('Final Score:', curr_score)
    print('Final Solution (Binary):', ''.join(map(str, curr_solution)))
    print('Running Duration:', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # Read data
    graph = read_nxgraph('./data/gset/gset_14.txt')
    # Run RL-based greedy algorithm
    num_steps = 100
    init_solution = [random.randint(0,1) for _ in range(graph.number_of_nodes())]
    rl_score, rl_solution, rl_scores = greedy_maxcut_rl(init_solution, num_steps, graph)


