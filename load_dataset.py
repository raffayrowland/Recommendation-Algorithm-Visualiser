import os
from datasets import load_dataset
from database import add_metadata

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

user_embeddings = load_dataset(
    "parquet",
    data_files="dataset/users/user-*.parquet",
    split="train",
)

item_embeddings = load_dataset(
    "parquet",
    data_files="dataset/items/item-*.parquet",
    split="train",
)

for i in range(10):
    print((metadata[i]["track_id"],
           metadata[i]["ISRC"][0],
           metadata[i]["track_name"][0],
           metadata[i]["artist_name"][0],
           metadata[i]["tag_list"])
          )

    add_metadata(metadata[i]["track_id"],
                 metadata[i]["ISRC"][0],
                 metadata[i]["track_name"][0],
                 metadata[i]["artist_name"][0],
                 metadata[i]["tag_list"])