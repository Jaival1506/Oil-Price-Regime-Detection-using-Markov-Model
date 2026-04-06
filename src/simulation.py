import numpy as np

def simulate_path(matrix, start_state, steps=15):
    states = matrix.columns.tolist()
    current = start_state
    path = [current]

    for _ in range(steps):
        probs = matrix.loc[current].values
        current = np.random.choice(states, p=probs)
        path.append(current)

    return path


def simulate_multiple_paths(matrix, start_state, steps=15, n_simulations=100):
    states = matrix.columns.tolist()
    all_paths = []

    for _ in range(n_simulations):
        current = start_state
        path = []

        for _ in range(steps):
            probs = matrix.loc[current].values
            current = np.random.choice(states, p=probs)
            path.append(current)

        all_paths.append(path)

    return all_paths