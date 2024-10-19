import numpy as np


# Define the generative model (mean and std dev)
def mean(x):
    return x - 2


def std_dev(x):
    return np.abs(x) + 1


# Define Variational Free Energy
def vfe(current_state, desired_state):
    return 0.5 * ((current_state - desired_state) ** 2) + np.log(std_dev(desired_state))


# Run Active Inference
current_state = 5.0
desired_state = -1.0

for _ in range(1000):
    action = current_state + np.random.normal(scale=0.1)
    vfe_val = vfe(action, desired_state)
    print(f"Action: {action:.2f}, VFE: {vfe_val:.4f}")
    current_state = action
