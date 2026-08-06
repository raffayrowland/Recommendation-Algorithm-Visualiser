import numpy as np
import pandas as pd
import joblib
import umap
import os
from database import get_info_for_visualisation


def build_vector_space(n, alpha):
    os.makedirs(f"spaces/{alpha}/{n}", exist_ok=True)
    points_path = f"spaces/{alpha}/{n}/points.pkl"
    reducer_path = f"spaces/{alpha}/{n}/reducer.pkl"

    if os.path.exists(points_path):
        return pd.read_pickle(points_path)

    points = get_info_for_visualisation(n, alpha)

    track_ids, track_names, artist_names, isrcs, combined_embeddings = map(
        list, zip(*points)
    )
    combined_embeddings = np.stack(
        [embedding.to_numpy() for embedding in combined_embeddings]
    )

    reducer = umap.UMAP(
        n_components=3,
        metric="euclidean",
        n_neighbors=30,
        min_dist=0.1,
        random_state=1
    ).fit(combined_embeddings)
    reduced_embeddings = reducer.embedding_

    songs_and_embeddings = pd.DataFrame({
        "track_id": track_ids,
        "track_name": track_names,
        "artist_name": artist_names,
        "isrc": isrcs,
        "combined_embedding": list(combined_embeddings),
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "z": reduced_embeddings[:, 2],
    })

    songs_and_embeddings.to_pickle(points_path)
    joblib.dump(reducer, reducer_path)

    return songs_and_embeddings


def add_additional_points(points, existing_space, n, alpha):
    reducer_path = f"spaces/{alpha}/{n}/reducer.pkl"

    neighbour_ids = [point[0] for point in points]
    existing_ids = set(existing_space["track_id"])
    missing_points = [point for point in points if point[0] not in existing_ids]

    existing_neighbours = existing_space[
        existing_space["track_id"].isin(neighbour_ids)
    ]

    if not missing_points:
        return (
            existing_neighbours.set_index("track_id")
            .loc[neighbour_ids]
            .reset_index()
        )

    if not os.path.exists(reducer_path):
        raise FileNotFoundError(f"Reducer not found: {reducer_path}")

    track_ids, track_names, artist_names, isrcs, combined_embeddings = map(
        list, zip(*missing_points)
    )
    combined_embeddings = np.stack(
        [embedding.to_numpy() for embedding in combined_embeddings]
    )

    reducer = joblib.load(reducer_path)
    new_coordinates = reducer.transform(combined_embeddings)

    additional_points = pd.DataFrame({
        "track_id": track_ids,
        "track_name": track_names,
        "artist_name": artist_names,
        "isrc": isrcs,
        "combined_embedding": list(combined_embeddings),
        "x": new_coordinates[:, 0],
        "y": new_coordinates[:, 1],
        "z": new_coordinates[:, 2],
    })

    all_neighbours = pd.concat(
        [existing_neighbours, additional_points], ignore_index=True
    )
    return (
        all_neighbours.set_index("track_id")
        .loc[neighbour_ids]
        .reset_index()
    )
