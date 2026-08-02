import os
from itertools import islice
from datasets import load_dataset
import numpy as np
from database import get_connection
from calculate_combined import *

connection = get_connection()
BATCH_SIZE = 10_000

# Batch generator
def batches(items):
    iterator = iter(items)
    while batch := list(islice(iterator, BATCH_SIZE)):
        yield batch

#os.makedirs("dataset/metadata", exist_ok=True)
#os.makedirs("dataset/users", exist_ok=True)
#os.makedirs("dataset/items", exist_ok=True)
#os.makedirs("dataset/clap", exist_ok=True)
#os.makedirs("dataset/mpd", exist_ok=True)
#
#metadata = load_dataset(
#    "parquet",
#    data_files="dataset/metadata/train-*.parquet",
#    split="train",
#)
#
## Insert all metadata rows
#count = 0
#with connection.cursor() as cursor:
#    for batch in batches(metadata):
#        rows = []
#        for item in batch:
#            track_name = item["track_name"]
#            if not track_name:
#                count += 1
#                continue
#
#            isrcs = item["ISRC"]
#            artist_name = item["artist_name"]
#            rows.append(
#                (
#                    item["track_id"],
#                    isrcs[0] if isrcs else None,
#                    track_name,
#                    artist_name,
#                    item["tag_list"],
#                    " ".join((*track_name, *artist_name)),
#                )
#            )
#
#        if not rows:
#            continue
#
#        cursor.executemany(
#            """
#            INSERT INTO metadata (
#                track_id, ISRC, track_name, artist_name, tag_list, search_document
#            )
#                VALUES (%s, %s, %s, %s, %s, to_tsvector('simple', %s))
#                ON CONFLICT (track_id) DO NOTHING
#            """,
#            rows,
#        )
#
#        count += len(rows)
#        if count % 10000 == 0:
#            print(f"\r{count} metadata rows inserted", end="")
#
#print(f"\r{count} metadata rows inserted")
#connection.commit()
#
#clap_embeddings = load_dataset(
#    "parquet",
#    data_files="dataset/clap/train-*.parquet",
#    split="train",
#)
#
## Insert all CLAP embeddings
#count = 0
#with connection.cursor() as cursor:
#    for batch in batches(clap_embeddings):
#        rows = [(item["id"], item["embedding"], item["id"]) for item in batch]
#
#        cursor.executemany(
#            """
#            INSERT INTO clap_embeddings (track_id, embedding)
#                SELECT %s, %s
#                WHERE EXISTS (
#                    SELECT 1 FROM metadata WHERE track_id = %s
#                )
#                ON CONFLICT (track_id) DO NOTHING
#            """,
#            rows,
#        )
#
#        count += len(rows)
#        if count % 10000 == 0:
#            print(f"\r{count} clap rows inserted", end="")
#
#print(f"\r{count} clap rows inserted")
#connection.commit()
#
#item_embeddings = load_dataset(
#    "parquet",
#    data_files="dataset/items/item-*.parquet",
#    split="train",
#)
#
## Insert all cf-bpr embeddings
#count = 0
#with connection.cursor() as cursor:
#    for batch in batches(item_embeddings):
#        rows = [(item["id"], item["embedding"], item["id"]) for item in batch]
#
#        cursor.executemany(
#            """
#            INSERT INTO cf_bpr (track_id, embedding)
#                SELECT %s, %s
#                WHERE EXISTS (
#                    SELECT 1 FROM metadata WHERE track_id = %s
#                )
#                ON CONFLICT (track_id) DO NOTHING
#            """,
#            rows,
#        )
#
#        count += len(rows)
#        if count % 10000 == 0:
#            print(f"\r{count} cfbpr rows inserted", end="")
#
#print(f"\r{count} cfbpr rows inserted")
#connection.commit()
#
## Delete rows that do not appear in all three databases
#with connection.cursor() as cursor:
#    cursor.execute(
#        """
#        CREATE TEMP TABLE common_track_ids
#        ON COMMIT DROP
#        AS
#        SELECT m.track_id
#        FROM metadata AS m
#        INNER JOIN clap_embeddings AS c
#            ON c.track_id = m.track_id
#        INNER JOIN cf_bpr AS b
#            ON b.track_id = m.track_id;
#
#        DELETE FROM clap_embeddings AS c
#        WHERE NOT EXISTS (
#            SELECT 1
#            FROM common_track_ids AS common
#            WHERE common.track_id = c.track_id
#        );
#
#        DELETE FROM cf_bpr AS b
#        WHERE NOT EXISTS (
#            SELECT 1
#            FROM common_track_ids AS common
#            WHERE common.track_id = b.track_id
#        );
#
#        DELETE FROM metadata AS m
#        WHERE NOT EXISTS (
#            SELECT 1
#            FROM common_track_ids AS common
#            WHERE common.track_id = m.track_id
#        );
#        """
#    )
#print("Deleted rows that do not appear in all three databases")
#connection.commit()


mpd = load_dataset(
    "parquet",
    data_files="dataset/mpd/train-*.parquet",
    split="train",
)

updates = {}
count = 0
for playlist in mpd:
    for track in playlist["track_ids"]:
        if track not in updates:
            updates[track] = 1

        else:
            updates[track] += 1

    count += 1
    if count % 50_000 == 0:
        print(f"\r{count} playlists processed", end="")

print(f"\r{count} playlists processed, {len(updates)} unique tracks")

count = 0
with connection.cursor() as cursor:
    for batch in batches(list(updates.items())):
        track_ids = [track_id for track_id, _ in batch]
        occurrences = [n for _, n in batch]

        cursor.execute(
            """
            UPDATE metadata AS m
            SET mpd_occurrences = u.occurrences 
            FROM unnest(%s::text[], %s::int[]) AS u(track_id, occurrences)
            WHERE m.track_id = u.track_id
            """, (track_ids, occurrences),
        )

        count += len(batch)
        if count % 1000 == 0:
            print(f"\r{count} occurrences updated", end="")

print(f"\r{count} occurrences updated")
connection.commit()

## Calculate and insert combined embeddings for alphas 0, 0.25, 0.5, 0.75, and 1
#count = 0
#with connection.cursor() as cursor:
#    all_track_ids = cursor.execute("SELECT track_id FROM metadata").fetchall()
#
#    for batch in batches(all_track_ids):
#        track_ids = [track_id[0] for track_id in batch]
#
#        cfbpr_clap = cursor.execute(
#            """
#            SELECT c.track_id,
#                   c.embedding AS clap_embedding,
#                   b.embedding AS cf_bpr_embedding
#            FROM clap_embeddings AS c
#                     JOIN cf_bpr AS b ON b.track_id = c.track_id
#            WHERE c.track_id = ANY (%s)
#            """,
#            (track_ids,),
#        ).fetchall()
#
#        rows = []
#        for track_id, clap_embedding, cfbpr_embedding in cfbpr_clap:
#            clap = clap_embedding.to_numpy()
#            cfbpr = cfbpr_embedding.to_numpy()
#            cfbpr, clap = normalise_vectors(cfbpr, clap)
#
#            rows.append((
#                track_id,
#                calculate_combined(cfbpr, clap, 0),
#                calculate_combined(cfbpr, clap, 0.25),
#                calculate_combined(cfbpr, clap, 0.5),
#                calculate_combined(cfbpr, clap, 0.75),
#                calculate_combined(cfbpr, clap, 1),
#            ))
#
#        cursor.executemany(
#            """
#            INSERT INTO combined_embeddings (track_id, emb_000, emb_025, emb_050, emb_075, emb_100)
#            VALUES (%s, %s, %s, %s, %s, %s)
#            ON CONFLICT (track_id) DO NOTHING
#            """, rows
#        )
#        count += len(rows)
#
#        if count % 10000 == 0:
#            print(f"{count} combined embeddings inserted\r", end="")
#
#print(f"{count} combined embeddings inserted")
#connection.commit()
#
## Build indexes to speed up NN and text search
#with connection.cursor() as cursor:
#    cursor.execute(
#        """
#        CREATE INDEX combined_embeddings_emb_000_hnsw_idx
#            ON combined_embeddings USING hnsw (emb_000 vector_cosine_ops);
#        CREATE INDEX combined_embeddings_emb_025_hnsw_idx
#            ON combined_embeddings USING hnsw (emb_025 vector_cosine_ops);
#        CREATE INDEX combined_embeddings_emb_050_hnsw_idx
#            ON combined_embeddings USING hnsw (emb_050 vector_cosine_ops);
#        CREATE INDEX combined_embeddings_emb_075_hnsw_idx
#            ON combined_embeddings USING hnsw (emb_075 vector_cosine_ops);
#        CREATE INDEX combined_embeddings_emb_100_hnsw_idx
#            ON combined_embeddings USING hnsw (emb_100 vector_cosine_ops);
#        CREATE INDEX metadata_search_document_gin_idx
#            ON metadata USING gin (search_document);
#        """
#    )
#print("Created indexes")
#
#connection.commit()
#connection.close()
#