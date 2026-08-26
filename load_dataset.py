import os
from itertools import islice
from datasets import load_dataset
import numpy as np
from psycopg import sql
import time
from database import get_connection, normalise_search_text
from calculate_combined import get_embeddings

BATCH_SIZE = 10_000
START_TIME = time.time()

# Batch generator
def batches(items, batch_size=BATCH_SIZE):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch

os.makedirs("dataset/metadata", exist_ok=True)
os.makedirs("dataset/items", exist_ok=True)
os.makedirs("dataset/clap", exist_ok=True)
os.makedirs("dataset/mpd", exist_ok=True)
os.makedirs("dataset/attributes", exist_ok=True)
os.makedirs("dataset/lyrics", exist_ok=True)

data_sources = {
    "clap": "dataset/clap/*.parquet",
    "collab": "dataset/items/*.parquet",
    "lyric": "dataset/lyrics/*.parquet",
    "attributes": "dataset/attributes/*.parquet",
}

def load_embeddings(source):
    connection = get_connection()
    dataset = load_dataset(
        "parquet",
        data_files=data_sources[source],
        split="train",
    )

    count = 0

    with connection.transaction(), connection.cursor() as cursor:
        for batch in batches(dataset):
            rows = [(item["id"], item["embedding"]) for item in batch]

            column = sql.Identifier(source).as_string(connection)

            query = f"""
                INSERT INTO track_embeddings (track_id, {column})
                VALUES (%s, %s)
                ON CONFLICT (track_id) DO UPDATE
                SET {column} = EXCLUDED.{column}
            """

            cursor.executemany(query, rows)

            count += len(rows)
            print(f"\r{count} {source} rows inserted" if count % 10000 == 0 else "", end="")

        print(f"\r{count} {source} rows inserted\n")

    connection.close()


def insert_metadata():
    metadata = load_dataset(
        "parquet",
        data_files="dataset/metadata/*.parquet",
        split="train",
    )

    count = 0
    with get_connection() as connection, connection.cursor() as cursor:
        for batch in batches(metadata, batch_size=100_000):
            rows = []

            for item in batch:
                if not item["track_name"] or not item["artist_name"] or not item["ISRC"]:
                    continue

                track_search = normalise_search_text(item["track_name"][0])
                artist_search = normalise_search_text(item["artist_name"][0])

                rows.append(
                    (
                        item["track_id"],
                        item["ISRC"][0],
                        item["track_name"][0],
                        item["artist_name"][0],
                        f"{track_search} {artist_search}".strip(),
                    )
                )

            cursor.executemany(
                """
                INSERT INTO tracks
                    (track_id, ISRC, track_name, artist_name, search_text)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (track_id) DO NOTHING
                """, rows,
            )
            count += len(rows)
            print(f"\r{count} metadata rows inserted", end="")

        connection.commit()


def delete_incomplete_data():
    with get_connection() as connection, connection.cursor() as cursor:
        # Delete rows which are missing one or more embedding
        print("Removing embedding rows that are missing one or more embedding")
        cursor.execute(
            """
            DELETE FROM track_embeddings
            WHERE collab IS NULL
               OR clap IS NULL
               OR lyric IS NULL
               OR attributes IS NULL;
            """
        )

        # Delete rows which do not have any embeddings
        print("Removing metadata rows which do not have corresponding embeddings")
        cursor.execute(
            """
            DELETE FROM tracks
            WHERE NOT EXISTS (
                SELECT 1
                FROM track_embeddings
                WHERE track_embeddings.track_id = tracks.track_id
            );
            """
        )

        # Delete embedding rows that do not have any metadata
        print("Removing embedding rows that do not have any metadata")
        cursor.execute(
            """
            DELETE FROM track_embeddings
            WHERE NOT EXISTS (
                SELECT 1
                FROM tracks
                WHERE tracks.track_id = track_embeddings.track_id
            );
            """
        )
        connection.commit()


def update_mpd_counts():
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
        print(f"\r{count} playlists processed" if count % 50_000 == 0 else "", end="")

    with get_connection() as connection, connection.cursor() as cursor:
        for batch in batches(updates.items()):
            track_ids, occurrences = map(list, zip(*batch))

            cursor.execute(
                """
                UPDATE tracks AS t
                SET mpd_occurrences = u.occurrences
                FROM unnest(%s::text[], %s::int[]) AS u(track_id, occurrences)
                WHERE t.track_id = u.track_id
                """,
                (track_ids, occurrences),
            )

        connection.commit()


def insert_combined_embeddings():
    with get_connection() as connection, connection.cursor() as cursor:
        track_ids = cursor.execute(
            """
            SELECT track_id
            FROM track_embeddings
            """
        ).fetchall()

        count = 0
        for batch in batches(track_ids):
            ids = [row[0] for row in batch]
            rows = cursor.execute(
                """
                SELECT track_id, collab, clap, lyric, attributes
                FROM track_embeddings
                WHERE track_id = ANY(%s)
                """, (ids,)
            ).fetchall()

            track_ids = [row[0] for row in rows]
            collabs = np.stack([row[1].to_numpy() for row in rows]).astype(
                np.float32, copy=False
            )
            claps = np.stack([row[2].to_numpy() for row in rows]).astype(
                np.float32, copy=False
            )
            lyrics = np.stack([row[3].to_numpy() for row in rows]).astype(
                np.float32, copy=False
            )
            attributes = np.stack([row[4].to_numpy() for row in rows]).astype(
                np.float32, copy=False
            )

            for modality in (collabs, claps, lyrics, attributes):
                norms = np.linalg.norm(modality, axis=1, keepdims=True)
                modality /= np.maximum(norms, 1e-12)

            embeddings = get_embeddings(collabs, claps, lyrics, attributes)
            track_id_embedding = list(zip(track_ids, embeddings))
            count += len(track_id_embedding)

            with connection.cursor() as insert_cursor:
                insert_cursor.executemany(
                    """
                    INSERT INTO combined_embedding (track_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (track_id) DO UPDATE SET embedding = EXCLUDED.embedding
                    """, track_id_embedding
                )

            print(f"\rInserted {count} rows", end="")

        connection.commit()


def build_index():
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL maintenance_work_mem = '6GB'")
        cursor.execute("SET LOCAL max_parallel_maintenance_workers = 8")
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS combined_embedding_hnsw_cosine_idx
        ON combined_embedding
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)

        print("Created combined embedding index")

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS tracks_search_text_gist_idx
        ON tracks
        USING GIST (search_text gist_trgm_ops);
        """)

        print("Created gist search index")

insert_metadata()

delete_incomplete_data()

update_mpd_counts()

insert_combined_embeddings()

build_index()
