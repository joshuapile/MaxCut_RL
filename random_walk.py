# compared methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
import copy
import time
import networkx as nx
import numpy as np
from typing import List, Union
import random
from util import read_nxgraph
from util import obj_maxcut


import sys
'''
Random Walk on MaxCut
Random Walk will randomly flip a number of of a current binary vector, then select the best progression
to approach a solution. A neighbor function will define the solution-node's neighbors through flipping
bits. Then, a select function will select a solution-node from the neighbor set.
'''

def random_walk(init_solution: Union[List[int], np.array], num_steps: int, max_num_flips: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('random_walk')
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    init_score = obj_maxcut(init_solution, graph)
    num_nodes = len(curr_solution)
    scores = []
    nodes = list(range(num_nodes))
    if max_num_flips > num_nodes:
        max_num_flips = num_nodes
    for i in range(num_steps):
        # select nodes randomly
        traversal_scores = []
        traversal_solutions = []
        #Loop to alter neighbors (Neighbor function)
        for j in range(1, max_num_flips + 1):
            selected_nodes = random.sample(nodes, j)
            new_solution = copy.deepcopy(curr_solution)
            new_solution = np.array(new_solution)
            new_solution[selected_nodes] = (new_solution[selected_nodes] + 1) % 2
            new_solution = new_solution.tolist()
            # calc the obj
            new_score = obj_maxcut(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        best_traversal_score = max(traversal_scores)
        index = traversal_scores.index(best_traversal_score)
        best_traversal_solution = traversal_solutions[index]
        #Determine if the proposed new solution is an improvement from the current accepted solution (Select function)
        if len(scores) == 0 or (len(scores) >= 1 and best_traversal_score >= scores[-1]):
            curr_solution = best_traversal_solution
            scores.append(best_traversal_score)
    score = max(scores)
    print("score, init_score of random_walk", score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return score, curr_solution, scores


if __name__ == '__main__':
    # read data
    # graph1 = read_as_networkx_graph('data/gset_14.txt')
    graph = read_nxgraph('./data/gset/gset_22.txt')

    # run alg
    # init_solution = [1, 0, 1, 0, 1]
    init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))
    rw_score, rw_solution, rw_scores = random_walk(init_solution=init_solution, num_steps=1000, max_num_flips=20, graph=graph)

'''
Disadvantage of Random Walk:

Random Walk suffers from noise due to statistical errors. Each step of random walk relies on a probablistic choice that will move
the algorithm towards the final answer. The distribution of end steps from random walk results in a Gaussian distribution. The 
standard deviation experienced by this distribution is sqrt{N}, where N is the number of steps taken. This highlights the chance
of error to occur when performing Random Walk.

Use Cases of Random Walk:

Random Walk is best utilized in problems that are random to begin with. Such examples include quantum walk, stock market analysis,
or thermodynamics. Each of these examples must consider the phemonenon of randomness to accurately portray a solution. In these
problems, the statistical error that can be expected from the algorithm is welcomed to increase the likelihood a valid solution
will be found.
'''


