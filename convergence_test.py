import numpy as np
import matplotlib.pyplot as plt

# System parameters
p = 0.8  # Channel transition probability (G -> G)
q = 0.6  # Channel transition probability (B -> G)
epsilon = 0.2  # Packet reception probability in state B
MAX_AGE = 10  # Maximum AoI value to consider

def generate_states(T):
    return [(t, a, c) for t in range(T+1)
                       for a in range(1, MAX_AGE+1)
                       for c in ['G', 'B']]

# Action space
ACTIONS = [0, 1]  # 0: idle, 1: transmit

# Channel transition probability matrix
P = np.array([[p, 1-p], [1-q, q]])

def transition_probability(current_state, action, next_state):
    t, a, c = current_state
    t_next, a_next, c_next = next_state

    if action == 0:  # Idle
        if c == c_next:
            if a_next == min(a + 1, MAX_AGE):
                return 1
        return 0

    else:  # Transmit
        if c == 'G':
            if c_next == 'G' and a_next == 1:
                return p
            elif c_next == 'B' and a_next == 1:
                return 1 - p
        else:  # c == 'B'
            if c_next == 'G' and a_next == 1:
                return epsilon * (1 - q)
            elif c_next == 'B' and a_next == 1:
                return epsilon * q
            elif c_next == 'G' and a_next == min(a + 1, MAX_AGE):
                return (1 - epsilon) * (1 - q)
            elif c_next == 'B' and a_next == min(a + 1, MAX_AGE):
                return (1 - epsilon) * q

    return 0

def backward_induction(T):
    STATES = generate_states(T)
    value_function = {s: 0 for s in STATES}
    policy = {s: None for s in STATES}

    for t in range(T, -1, -1):
        for a in range(1, MAX_AGE+1):
            for c in ['G', 'B']:
                state = (t, a, c)

                if t == T:
                    value_function[state] = a
                    continue

                min_cost = float('inf')
                optimal_action = None

                for action in ACTIONS:
                    cost = a
                    for next_state in STATES:
                        t_next, a_next, c_next = next_state
                        if t_next == t + 1:
                            prob = transition_probability(state, action, next_state)
                            cost += prob * value_function[next_state]

                    if cost < min_cost:
                        min_cost = cost
                        optimal_action = action

                value_function[state] = min_cost
                policy[state] = optimal_action

    return value_function, policy

def simulate(policy, initial_state, T):
    state = initial_state
    total_aoi = 0
    channel_state = 'G'

    for t in range(T+1):
        t_curr, a_curr, c_curr = state
        action = policy[state]

        total_aoi += a_curr

        if action == 1:  # Transmit
            if c_curr == 'G':
                a_next = 1
            else:  # c_curr == 'B'
                a_next = 1 if np.random.rand() < epsilon else min(a_curr + 1, MAX_AGE)

            c_next = np.random.choice(['G', 'B'], p=P[0 if channel_state == 'G' else 1])
            channel_state = c_next
        else:  # Idle
            a_next = min(a_curr + 1, MAX_AGE)
            c_next = c_curr

        state = (t+1, a_next, c_next)

    return total_aoi / (T+1)

# Compute average AoI for different T values
#T_values = [1000, 2000, 3000, 4000, 5000, 6000]
T_values = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000]  # Extended T values
avg_aoi_values = []
num_simulations = 1000  # Increased number of simulations

for T in T_values:
    value_function, policy = backward_induction(T)
    avg_aoi = np.mean([simulate(policy, (0, 1, 'G'), T) for _ in range(num_simulations)])
    avg_aoi_values.append(avg_aoi)
    print(f"T = {T}, Average AoI: {avg_aoi:.4f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(T_values, avg_aoi_values, marker='o')
plt.xlabel('Time Horizon (T)')
plt.ylabel('Average AoI')
plt.title('Average AoI vs Time Horizon')
plt.grid(True)
plt.xscale('log')  # Use logarithmic scale for x-axis
plt.show()