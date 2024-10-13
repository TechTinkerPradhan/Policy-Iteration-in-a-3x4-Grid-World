# Policy Iteration in a 3x4 Grid World


## Problem Overview
This project implements **Policy Iteration** in a 3x4 grid world using **Markov Decision Processes (MDP)** to find an optimal policy that maximizes cumulative rewards. The project involves alternating between policy evaluation and policy improvement steps until the policy converges.

### Problem Components:
- **States**: Different cells in the 3x4 matrix.
- **Actions**: UP, DOWN, LEFT, RIGHT.
- **Rewards**:
  - Terminal state rewards: -1 or +1.
  - For each non-terminal action: -0.04.
- **Transition Probabilities**:
  - Moving in the desired direction: 0.8
  - Moving left or right: 0.1 each.
  
### Key Equations:
- **Bellman Equation** for policy evaluation:

  \[
  V(s) = \sum_{s'} P(s' \mid s, \pi(s)) \left[ R(s, s') + \gamma V(s') \right]
  \]

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/policy-iteration-gridworld.git
