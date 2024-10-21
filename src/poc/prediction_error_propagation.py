import numpy as np


def init_state_motion():
    x = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)

    x_pred = np.zeros(3)
    v_pred = np.zeros(3)
    a_pred = np.zeros(3)

    return x, v, a, x_pred, v_pred, a_pred


def init_neuronal_populations(pop_size):
    x_Pop = np.random.rand(pop_size, 3)
    v_Pop = np.random.rand(pop_size, 3)
    a_Pop = np.random.rand(pop_size, 3)

    return x_Pop, v_Pop, a_Pop


def propagate_prediction_errors(
    x, v, a, x_pred, v_pred, a_pred, sensory_data, x_Pop, v_Pop, a_Pop
):
    ε_x = x - sensory_data["pos"]
    ε_v = v - sensory_data["vel"]
    ε_a = a - sensory_data["acc"]

    # print(f"Shape of ε_x: {ε_x.shape}, Shape of x_Pop: {x_Pop.shape}")

    x_pred += learning_rate * ε_x
    v_pred += learning_rate * ε_v
    a_pred += learning_rate * ε_a

    x_Pop += learning_rate * (ε_x - ε_x.mean())[np.newaxis, :] * x_Pop
    v_Pop += learning_rate * (ε_v - ε_v.mean())[np.newaxis, :] * v_Pop
    a_Pop += learning_rate * (ε_a - ε_a.mean())[np.newaxis, :] * a_Pop

    return x_pred, v_pred, a_pred, x_Pop, v_Pop, a_Pop, ε_x, ε_v, ε_a


def update_state(x, v, a, x_pred, v_pred, a_pred, x_Pop, v_Pop, a_Pop):
    μ_x = x_Pop.mean(axis=0)
    σ_x = np.std(x_Pop, axis=0)

    μ_v = v_Pop.mean(axis=0)
    σ_v = np.std(v_Pop, axis=0)

    μ_a = a_Pop.mean(axis=0)
    σ_a = np.std(a_Pop, axis=0)

    x += dt * μ_v + dt**2 * μ_a
    v += dt * μ_a
    a = np.random.normal(loc=μ_a, scale=σ_a)

    return x, v, a


# Simulation parameters
dt = 0.1
learning_rate = 0.05

x, v, a, x_pred, v_pred, a_pred = init_state_motion()
x_Pop, v_Pop, a_Pop = init_neuronal_populations(pop_size=10)

for t in range(1000):
    sensory_data = {
        "pos": np.random.normal(loc=x_pred, scale=0.1),
        "vel": np.random.normal(loc=v_pred, scale=0.1),
        "acc": np.random.normal(loc=a_pred, scale=0.1),
    }

    # print(
    #     f"Shape of x: {x.shape}, Shape of sensory_data['pos']: {sensory_data['pos'].shape}"
    # )

    x_pred, v_pred, a_pred, x_Pop, v_Pop, a_Pop, ε_x, ε_v, ε_a = (
        propagate_prediction_errors(
            x, v, a, x_pred, v_pred, a_pred, sensory_data, x_Pop, v_Pop, a_Pop
        )
    )

    x, v, a = update_state(x, v, a, x_pred, v_pred, a_pred, x_Pop, v_Pop, a_Pop)

    print(
        f"Time: {t:.2f}, State: [{x[0]:.4f}, {v[0]:.4f}, {a[0]:.4f}], Prediction Errors: [{ε_x.mean():.4f}, {ε_v.mean():.4f}, {ε_a.mean():.4f}]"
    )
