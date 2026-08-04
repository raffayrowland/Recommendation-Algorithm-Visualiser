import numpy as np
import pandas as pd
import umap
import os
from database import get_info_for_visualisation


def build_vector_space(n, alpha):
    os.makedirs("spaces", exist_ok=True)

    if os.path.exists(f"spaces/space_{n}_{alpha}.pkl"):
        return pd.read_pickle(f"spaces/space_{n}_{alpha}.pkl")

    points = get_info_for_visualisation(n, alpha)

    track_ids, track_names, artist_names, combined_embeddings = map(list, zip(*points))
    combined_embeddings = np.stack(
        [embedding.to_numpy() for embedding in combined_embeddings]
    )

    reduced_embeddings = umap.UMAP(
        n_components=3,
        metric="euclidean",
        n_neighbors=30,
        min_dist=0.1,
    ).fit_transform(combined_embeddings)

    songs_and_embeddings = pd.DataFrame({
        "track_id": track_ids,
        "track_name": track_names,
        "artist_name": artist_names,
        "combined_embedding": list(combined_embeddings),
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "z": reduced_embeddings[:, 2],
    })

    songs_and_embeddings.to_pickle(f"spaces/space_{n}_{alpha}.pkl")

    return songs_and_embeddings