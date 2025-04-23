# dqn_maxcut_refactored.py
"""
Refactored Deep Q‑Learning implementation for the Max‑Cut problem.

Key improvements over the original script
-----------------------------------------
* **Modular design** – separates environment, replay buffer, agent, and training
  logic to make experimentation easier.
* **Type hints & doc‑strings** – improve readability and editor support.
* **Dataclass hyper‑parameter bundle** – hyper‑parameters live in one place.
* **Vectorised replay buffer sampling** – faster mini‑batch construction.
* **Target‑network update with polyak averaging** – smoother learning.
* **Seed helper** – guarantees reproducibility with a single call.

"""
from __future__ import annotations

import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from util import obj_maxcut, read_nxgraph

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Fix all RNGs (Python/NumPy/PyTorch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Hyper‑parameters
# -----------------------------------------------------------------------------

@dataclass
class HParams:
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    lr: float = 1e-2
    batch_size: int = 64
    memory_size: int = 10_000
    target_update_freq: int = 10  # episodes
    tau: float = 1.0  # Polyak factor when copying the online net to the target net
    hidden_size: int = 64
    seed: int = 42


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Fixed‑size cyclic buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:  # noqa: ANN001
        self.buffer.append(Transition(*args))

    def __len__(self) -> int:  # noqa: D401
        return len(self.buffer)

    def sample(self, batch_size: int, device: torch.device) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        transposed = Transition(*zip(*batch))
        # Convert to tensors here for cleanliness
        states = torch.tensor(transposed.state, dtype=torch.float32, device=device)
        actions = torch.tensor(transposed.action, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(transposed.reward, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(transposed.next_state, dtype=torch.float32, device=device)
        dones = torch.tensor(transposed.done, dtype=torch.float32, device=device).unsqueeze(1)
        return Transition(states, actions, rewards, next_states, dones)


# -----------------------------------------------------------------------------
# Neural network
# -----------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Simple two‑layer perceptron returning Q‑values for two actions."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


# -----------------------------------------------------------------------------
# Max‑Cut environment helpers (feature extractor)
# -----------------------------------------------------------------------------

def node_features(node: int, sol: Sequence[int], g: nx.Graph) -> List[float]:
    """Return 4‑D feature vector for *node* under current solution *sol*."""
    neighbors = list(g.neighbors(node))
    curr = sol[node]
    opp = 1 - curr
    intra = sum(float(g[node][nbr].get("weight", 1.0)) for nbr in neighbors if sol[nbr] == curr)
    inter = sum(float(g[node][nbr].get("weight", 1.0)) for nbr in neighbors if sol[nbr] == opp)
    deg = g.degree[node]
    gain = inter - intra  # what cut value would change if flipped
    return [deg, intra, inter, gain]


# -----------------------------------------------------------------------------
# DQN Agent encapsulating behaviour
# -----------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, input_size: int, hparams: HParams, device: torch.device) -> None:
        self.hp = hparams
        self.device = device
        self.policy_net = QNetwork(input_size, hparams.hidden_size).to(device)
        self.target_net = QNetwork(input_size, hparams.hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hparams.lr)
        self.memory = ReplayBuffer(hparams.memory_size)
        self.loss_fn = nn.MSELoss()

        self.epsilon = hparams.epsilon_start

    # --------------------------- public API --------------------------- #
    def select_action(self, state: List[float]) -> int:
        if random.random() < self.epsilon:
            return random.choice((0, 1))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(s)
            return q_vals.argmax(dim=1).item()

    def push_transition(self, *args) -> None:  # noqa: ANN001
        self.memory.push(*args)

    def train_step(self) -> None:
        if len(self.memory) < self.hp.batch_size:
            return
        batch = self.memory.sample(self.hp.batch_size, self.device)

        q_values = self.policy_net(batch.state).gather(1, batch.action)
        with torch.no_grad():
            next_q = self.target_net(batch.next_state).max(dim=1, keepdim=True).values
            target = batch.reward + self.hp.gamma * next_q * (1 - batch.done)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_eps(self) -> None:
        self.epsilon = max(self.hp.epsilon_end, self.epsilon * self.hp.epsilon_decay)

    def sync_target_net(self) -> None:
        """Hard update (tau = 1) or Polyak averaging otherwise."""
        if self.hp.tau == 1.0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t.data.lerp_(p.data, self.hp.tau)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_dqn_maxcut(
    g: nx.Graph,
    num_episodes: int = 50,
    hparams: HParams | None = None,
) -> Tuple[int, List[int], List[int]]:
    """Return best score, solution and per‑episode scores."""
    hp = hparams or HParams()
    set_global_seed(hp.seed)
    device = get_device()

    agent = DQNAgent(input_size=4, hparams=hp, device=device)

    init_solution: List[int] = [random.randint(0, 1) for _ in g.nodes]

    best_score = obj_maxcut(init_solution, g)
    best_solution = init_solution[:]
    scores: List[int] = []

    for ep in range(1, num_episodes + 1):
        sol = best_solution[:]  # start each episode from current best (greedy‑ish)
        score = obj_maxcut(sol, g)

        for node in g.nodes:
            s = node_features(node, sol, g)
            a = agent.select_action(s)

            next_sol = sol[:]
            reward = 0.0
            done = 0.0  # no terminal condition yet
            if a == 1:  # flip
                next_sol[node] = 1 - next_sol[node]
                next_score = obj_maxcut(next_sol, g)
                reward = float(next_score - score)
                score = next_score
                sol = next_sol
            nxt_state = node_features(node, sol, g)
            agent.push_transition(s, a, reward, nxt_state, done)
            agent.train_step()

        # episode end – book‑keeping
        agent.update_eps()
        if ep % hp.target_update_freq == 0:
            agent.sync_target_net()

        scores.append(score)
        if score > best_score:
            best_score, best_solution = score, sol[:]

        print(f"Episode {ep:3d} | score = {score:>5d} | eps = {agent.epsilon:.3f}")

    return best_score, best_solution, scores


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = Path("./data/syn/syn_50.txt")
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    graph = read_nxgraph(DATA_PATH)
    EPISODES = 50

    t0 = time.perf_counter()
    best, sol, history = train_dqn_maxcut(graph, EPISODES)
    dur = time.perf_counter() - t0

    print("\n=== Training complete ===")
    print(f"Best cut value : {best}")
    print(f"Best solution  : {''.join(map(str, sol))}")
    print(f"Elapsed time   : {dur:.2f} seconds")
