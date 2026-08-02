import pandas as pd
import numpy as np
import plotly.graph_objects as go
import umap


def normalise_vectors(vectors):
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Avoid division by zero
    magnitudes[magnitudes == 0] = 1

    return vectors / magnitudes


songs = list(get_random_songs(10000))

track_ids = np.array([
    track_id for track_id, _, _ in songs
])

cfbpr_embeddings = np.stack([
    np.asarray(cfbpr.to_numpy(), dtype=float) for _, cfbpr, _ in songs
])

clap_embeddings = np.stack([
    np.asarray(clap.to_numpy(), dtype=float) for _, _, clap in songs
])

prop = 0.5

combined_embeddings = np.concatenate([
    np.sqrt(prop) * normalise_vectors(cfbpr_embeddings),
    np.sqrt(1 - prop) * normalise_vectors(clap_embeddings),
], axis=1)

reduced_embeddings = umap.UMAP(
    n_components=3,
    metric="euclidean",
    n_neighbors=30,
    min_dist=0.1,
    random_state=42,
).fit_transform(combined_embeddings)

songs_and_embeddings = pd.DataFrame({
    "track_id": track_ids,
    "x": reduced_embeddings[:, 0],
    "y": reduced_embeddings[:, 1],
    "z": reduced_embeddings[:, 2],
})

figure = go.Figure(
    data=[
        go.Scatter3d(
            x=songs_and_embeddings["x"],
            y=songs_and_embeddings["y"],
            z=songs_and_embeddings["z"],
            customdata=songs_and_embeddings[["track_id"]],
            mode="markers",
            marker={"size": 1.5, "opacity": 1},
            hovertemplate="Track ID: %{customdata[0]}<extra></extra>",
        )
    ]
)

figure.update_layout(
    title="Song recommendation vector space",
    scene={
        "xaxis_title": "",
        "yaxis_title": "",
        "zaxis_title": "",
    },
)

figure.show()
