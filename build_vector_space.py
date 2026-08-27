import numpy as np
import pandas as pd
import joblib
import umap
from pathlib import Path
from database import get_info_for_visualisation

PROJECT_ROOT = Path(__file__).resolve().parent
SPACES_DIR = PROJECT_ROOT / "spaces"


def build_vector_space(n):
    space_dir = SPACES_DIR / str(n)
    space_dir.mkdir(parents=True, exist_ok=True)
    points_path = space_dir / "points.pkl"
    reducer_path = space_dir / "reducer.pkl"

    if points_path.exists():
        return pd.read_pickle(points_path)

    points = get_info_for_visualisation(n)

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


def add_additional_points(points, existing_space, n):
    reducer_path = SPACES_DIR / str(n) / "reducer.pkl"

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

    if not reducer_path.exists():
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
