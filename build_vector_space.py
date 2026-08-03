import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from database import get_info_for_visualisation

points = get_info_for_visualisation(40000, 0.75)

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
            customdata=songs_and_embeddings[["track_name", "artist_name"]],
            mode="markers",
            marker={"size": 1.5, "opacity": 1},
            hovertemplate="Song: %{customdata[0]}<br>Artist: %{customdata[1]}<extra></extra>",
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
