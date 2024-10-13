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
Results
The policy iteration process alternates between evaluating a given policy and improving it. It converges when the policy no longer changes.
The output includes:
The optimal policy.
The final state-value function.
Graphs showing the convergence of the value function and how the values of specific states change over iterations.
Visual Outputs
Policy and State-Value Function over Iterations

Final Optimal Policy and State-Value Function

Convergence of the Value Function

Value of State (0,0) Over Iterations

Value of State (2,2) Over Iterations for Different Gamma

Challenges
1. Double Counting Terminal Rewards
Initially, terminal states caused double-counting of rewards, inflating the state values. The issue was resolved by excluding the future discounted value for terminal states.

2. Using Different Gamma Values
Varying the discount factor $\gamma$ changes the behavior of the policy, impacting the decision-making for short-term versus long-term rewards.
