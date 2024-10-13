import numpy as np
import matplotlib.pyplot as plt

# Grid world parameters
grid_rows = 3
grid_cols = 4
gamma = 0.9        # Discount factor
theta = 1e-6       # Convergence threshold

# Terminal states and their rewards
terminals = {
    (0, 3): +1.0,
    (1, 3): -1.0
}

# Obstacles (walls)
obstacles = [(1, 1)]

# Possible actions
actions = ['U', 'D', 'L', 'R']

# Action vectors for movement
action_vectors = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# Mapping for left and right turns
action_left = {'U': 'L', 'L': 'D', 'D': 'R', 'R': 'U'}
action_right = {'U': 'R', 'R': 'D', 'D': 'L', 'L': 'U'}

# Initial policy as given
initial_policy = {
    (2, 0): 'U',
    (2, 1): 'L',
    (2, 2): 'U',
    (2, 3): 'L',
    (1, 0): 'R',
    (1, 2): 'D',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'D',
}

# All states in the grid world
states = []
for i in range(grid_rows):
    for j in range(grid_cols):
        s = (i, j)
        if s not in obstacles and s not in terminals:
            states.append(s)

# Initialize the policy
policy = {}
for s in states:
    if s in initial_policy:
        policy[s] = initial_policy[s]
    else:
        policy[s] = np.random.choice(actions)  # Assign a random action for unspecified states

# Initialize the state-value function V(s) to zero, including terminal states
V = {}
for s in states:
    V[s] = 0.0
for s in terminals:
    V[s] = 0.0  # Terminal states have no future value

def is_valid_state(s):
    """Check if the state is within the grid and not an obstacle."""
    i, j = s
    return 0 <= i < grid_rows and 0 <= j < grid_cols and s not in obstacles

def get_transitions(s, a):
    """
    Get the list of possible transitions from state s when action a is taken.
    Returns a list of tuples: (probability, next_state, reward)
    """
    transitions = []
    if s in terminals:
        # No transitions from terminal states
        return transitions

    # Intended action and unintended stochastic actions
    action_probabilities = [
        (a, 0.8),
        (action_left[a], 0.1),
        (action_right[a], 0.1)
    ]

    for action, prob in action_probabilities:
        di, dj = action_vectors[action]
        next_state = (s[0] + di, s[1] + dj)
        # Check for collisions with walls or obstacles
        if not is_valid_state(next_state):
            next_state = s  # Agent stays in the same state

        # Get the reward for the move
        if next_state in terminals:
            reward = terminals[next_state]  # Immediate reward upon entering terminal state
        else:
            reward = -0.04  # Standard reward for non-terminal moves

        transitions.append((prob, next_state, reward))

    return transitions

def print_policy(policy, iteration=None):
    """Print the policy grid."""
    if iteration is not None:
        print(f"\nPolicy after iteration {iteration}:")
    else:
        print("\nPolicy:")
    for i in range(grid_rows):
        for j in range(grid_cols):
            s = (i, j)
            if s in terminals:
                print(f" {terminals[s]:+} ", end='')
            elif s in obstacles:
                print(" XX ", end='')
            else:
                action = policy[s]
                # Use arrows for better visualization
                action_symbols = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
                print(f" {action_symbols[action]} ", end='')
        print()
        
def print_value_function(V, iteration=None):
    """Print the state-value function grid."""
    if iteration is not None:
        print(f"\nState-Value Function after iteration {iteration}:")
    else:
        print("\nState-Value Function:")
    for i in range(grid_rows):
        for j in range(grid_cols):
            s = (i, j)
            if s in V:
                print(f"{V[s]:6.2f} ", end='')
            else:
                print("  XX   ", end='')
        print()

# Variables to store convergence data
delta_history = []
value_history = []

# Policy Iteration Algorithm
is_policy_stable = False
iteration = 0
while not is_policy_stable:
    iteration += 1
    # Policy Evaluation
    while True:
        delta = 0
        for s in states:
            v = V[s]
            action = policy[s]
            transitions = get_transitions(s, action)
            V[s] = sum([
                prob * (reward + gamma * V[next_state])
                if next_state not in terminals else prob * reward
                for prob, next_state, reward in transitions
            ])
            delta = max(delta, abs(v - V[s]))
        delta_history.append(delta)  # Record delta for plotting
        if delta < theta:
            break

    # storing value function for plotting
    value_history.append(V.copy())

    # Policy Improvement
    is_policy_stable = True
    for s in states:
        old_action = policy[s]
        action_values = {}
        for a in actions:
            transitions = get_transitions(s, a)
            action_value = sum([
                prob * (reward + gamma * V[next_state])
                if next_state not in terminals else prob * reward
                for prob, next_state, reward in transitions
            ])
            action_values[a] = action_value
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action
        if old_action != best_action:
            is_policy_stable = False

    # Print the policy and value function after each iteration
    print_policy(policy, iteration)
    print_value_function(V, iteration)

# Final output
print("\nOptimal Policy:")
print_policy(policy)
print("\nFinal State-Value Function:")
print_value_function(V)

# Plotting the convergence of delta
plt.figure(figsize=(10, 6))
plt.plot(delta_history)
plt.title('Convergence of Value Function')
plt.xlabel('Iterations')
plt.ylabel('Delta (Max Change in V)')
plt.yscale('log')
plt.grid(True)
plt.show()

# Optional: Plot the value of a specific state over iterations
state_to_track = (0, 0)  # You can choose any non-terminal state
state_values_over_time = [v[state_to_track] for v in value_history]

plt.figure(figsize=(10, 6))
plt.plot(state_values_over_time)
plt.title(f'Value of State {state_to_track} Over Iterations')
plt.xlabel('Iterations')
plt.ylabel(f'V({state_to_track})')
plt.grid(True)
plt.show()
