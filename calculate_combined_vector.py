import numpy as np


def normalise_vector(vector):
    if hasattr(vector, "to_numpy"):
        vector = vector.to_numpy()

    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)

    return vector / magnitude


def calculate_combined_vector(bpr, clap, prop):
    combined = np.concatenate([
        np.sqrt(prop) * normalise_vector(bpr),
        np.sqrt(1 - prop) * normalise_vector(clap),
    ])

    return combined
