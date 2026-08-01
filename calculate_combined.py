import numpy as np

def normalise_vectors(cfbpr, clap):
    cfbpr_norm = cfbpr / np.linalg.norm(cfbpr)
    clap_norm = clap / np.linalg.norm(clap)

    return cfbpr_norm, clap_norm

def calculate_combined(cfbpr, clap, alpha):
    combined = np.concatenate(
        [
            np.sqrt(alpha) * cfbpr,
            np.sqrt(1 - alpha) * clap,
        ]
    )

    return combined