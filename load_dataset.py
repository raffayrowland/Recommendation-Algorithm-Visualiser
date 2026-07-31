import os
from datasets import load_dataset
import numpy as np
from database import *

os.makedirs("dataset/metadata", exist_ok=True)
os.makedirs("dataset/users", exist_ok=True)
os.makedirs("dataset/items", exist_ok=True)
os.makedirs("dataset/clap", exist_ok=True)

metadata = load_dataset(
    "parquet",
    data_files="dataset/metadata/train-*.parquet",
    split="train",
)

clap_embeddings = load_dataset(
    "parquet",
    data_files="dataset/clap/train-*.parquet",
    split="train",
)

# user_embeddings = load_dataset(
#     "parquet",
#     data_files="dataset/users/user-*.parquet",
#     split="train",
# )

item_embeddings = load_dataset(
    "parquet",
    data_files="dataset/items/item-*.parquet",
    split="train",
)

def insert_metadata():
    print(metadata[0])
    metadata_rows = []
    for item in metadata:
        if item["ISRC"] == [] or item["track_name"] == [] or item["artist_name"] == []:
            continue

        metadata_rows.append(
            (
                item["track_id"],
                item["ISRC"][0],
                item["track_name"][0],
                item["artist_name"][0],
                item["tag_list"]
            )
        )

        if len(metadata_rows) >= 1_000_000:
            add_metadata_bulk(metadata_rows)
            print(f"Inserted {len(metadata_rows)} metadata")
            metadata_rows = []

    add_metadata_bulk(metadata_rows)
    print(f"Inserted {len(metadata_rows)} metadata rows\n\n")


def insert_clap_embeddings():
    print(clap_embeddings[0])
    clap_embeddings_rows = []
    for item in clap_embeddings:
        if item["id"] == "" or item["embedding"] == []:
            continue

        clap_embeddings_rows.append(
            (
                item["id"],
                item["embedding"]
            )
        )

        if len(clap_embeddings_rows) >= 100_000:
            add_clap_embeddings_bulk(clap_embeddings_rows)
            print(f"Inserted {len(clap_embeddings_rows)} clap embeddings")
            clap_embeddings_rows = []

    add_clap_embeddings_bulk(clap_embeddings_rows)
    print(f"Inserted {len(clap_embeddings_rows)} clap embeddings\n\n")


def insert_cfbpr_embeddings():
    print(item_embeddings[0])
    cfpbr_embeddings_rows = []

    for item in item_embeddings:
        if item["id"] == "" or item["embedding"] == []:
            continue

        cfpbr_embeddings_rows.append(
            (
                item["id"],
                item["embedding"]
            )
        )

        if len(cfpbr_embeddings_rows) >= 100_000:
            add_cfbpr_bulk(cfpbr_embeddings_rows)
            print(f"Inserted {len(cfpbr_embeddings_rows)} cfbpr embeddings")
            cfpbr_embeddings_rows = []

    add_cfbpr_bulk(cfpbr_embeddings_rows)
    print(f"Inserted {len(cfpbr_embeddings_rows)} cfbpr embeddings")


def store_combined_embeddings():
    def normalise_vectors(vectors):
        magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Avoid division by zero
        magnitudes[magnitudes == 0] = 1

        return vectors / magnitudes

    all_track_ids = get_all_track_ids()

    print(all_track_ids[0])

    for track_id in all_track_ids:
        cfbpr = get_cfbpr(track_id[0]).to_numpy()
        clap = get_clap(track_id[0]).to_numpy()

        norm_cfbpr = normalise_vectors(cfbpr)
        norm_clap = normalise_vectors(clap)

        combined_vectors = [track_id]
        for alpha in [0.25, 0.5, 0.75, 1]:
            combined = np.concatenate(
                np.sqrt(alpha) * norm_cfbpr,
                np.sqrt(1 - alpha) * norm_clap
            )
            combined_vectors.append(combined)

        add_combined(combined_vectors)


insert_metadata()

insert_clap_embeddings()

insert_cfbpr_embeddings()

store_combined_embeddings()