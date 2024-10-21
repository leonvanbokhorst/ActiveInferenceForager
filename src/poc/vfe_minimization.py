import numpy as np

# System parameters
m_x, m_y = 0.9, 0.95
σ_x, σ_y = 0.1, 0.2

# Initial beliefs (mean and variance for state)
μ, σ = 0, 1

# Temperature parameter
β = 0.5


def vfe(y_t, μ, σ):
    # Calculate VFE using current beliefs and observed state
    p = np.exp(-((y_t - m_y * μ) ** 2) / (2 * σ_y**2))
    vfe_val = -np.log(p) + β * ((μ**2 / σ**2) - np.log(σ**2))
    return vfe_val


def update_beliefs(y_t, μ, σ):
    # Update beliefs based on observed state and VFE minimization
    μ_next = (σ_y**2 * m_x * μ + σ_x**2 * y_t) / (σ_x**2 + σ_y**2)
    σ_next = np.sqrt((σ_x**2 * σ_y**2) / (σ_x**2 + σ_y**2))
    return μ_next, σ_next


def take_action(σ):
    # Simple action based on current uncertainty (i.e., variance)
    a_t = np.random.normal(scale=0.1) if σ > 0.5 else 0.0
    return a_t


# Simulation loop
for t in range(10):
    # Take action based on current uncertainty
    a_t = take_action(σ)

    # Generate next state based on action and system parameters
    x_t = np.random.normal(loc=m_x * μ + a_t, scale=np.sqrt(σ_x**2))

    # Observe next state with Gaussian noise
    y_t = np.random.normal(loc=m_y * x_t, scale=np.sqrt(σ_y**2))

    # Calculate VFE with observed state and current beliefs
    vfe_val = vfe(y_t, μ, σ)
    print(f"Action: {a_t:.2f}, VFE: {vfe_val:.4f}")

    # Update beliefs based on observed state and VFE minimization
    μ, σ = update_beliefs(y_t, μ, σ)
