import os
from itertools import islice
from datasets import load_dataset
from psycopg import sql
import time
from database import get_connection, normalise_search_text
from calculate_combined import *

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

    query = sql.SQL("""
    INSERT INTO track_embeddings (track_id, {col})
    VALUES (%s, %s)
    ON CONFLICT (track_id) DO UPDATE SET {col} = EXCLUDED.{col}
    """).format(col=sql.Identifier(source),)

    count = 0

    with connection.transaction(), connection.cursor() as cursor:
        for batch in batches(dataset):
            rows = [(item["id"], item["embedding"]) for item in batch]

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
                        track_search,
                        artist_search
                    )
                )

            cursor.executemany(
                """
                INSERT INTO tracks
                    (track_id, ISRC, track_name, artist_name, search_text, search_document)
                VALUES (%s, %s, %s, %s, %s,
                        setweight(to_tsvector('simple', %s), 'A') ||
                        setweight(to_tsvector('simple', %s), 'B'))
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

#for embedding_column, _ in data_sources.items():
#    load_embeddings(embedding_column)
#
#insert_metadata()
#
delete_incomplete_data()

# update_mpd_counts()

exit()

# Calculate and insert combined embeddings for alphas 0, 0.25, 0.5, 0.75, and 1
count = 0
with connection.cursor() as cursor:
    all_track_ids = cursor.execute("SELECT track_id FROM metadata").fetchall()

    for batch in batches(all_track_ids):
        track_ids = [track_id[0] for track_id in batch]

        cfbpr_clap = cursor.execute(
            """
            SELECT c.track_id,
                   c.embedding AS clap_embedding,
                   b.embedding AS cf_bpr_embedding
            FROM clap_embeddings AS c
                     JOIN cf_bpr AS b ON b.track_id = c.track_id
            WHERE c.track_id = ANY (%s)
            """,
            (track_ids,),
        ).fetchall()

        rows = []
        for track_id, clap_embedding, cfbpr_embedding in cfbpr_clap:
            clap = clap_embedding.to_numpy()
            cfbpr = cfbpr_embedding.to_numpy()
            cfbpr, clap = normalise_vectors(cfbpr, clap)

            rows.append((
                track_id,
                calculate_combined(cfbpr, clap, 0),
                calculate_combined(cfbpr, clap, 0.25),
                calculate_combined(cfbpr, clap, 0.5),
                calculate_combined(cfbpr, clap, 0.75),
                calculate_combined(cfbpr, clap, 1),
            ))

        cursor.executemany(
            """
            INSERT INTO combined_embeddings (track_id, emb_000, emb_025, emb_050, emb_075, emb_100)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO NOTHING
            """, rows
        )
        count += len(rows)

        if count % 10000 == 0:
            print(f"{count} combined embeddings inserted\r", end="")

print(f"{count} combined embeddings inserted")
connection.commit()

# Build indexes to speed up NN, text search, and popularity ordering
indexes = [
    (
        "combined_embeddings_emb_000_ivfflat_idx",
        """
        CREATE INDEX IF NOT EXISTS combined_embeddings_emb_000_ivfflat_idx
            ON combined_embeddings USING ivfflat (emb_000 vector_cosine_ops)
            WITH (lists = 1215)
        """,
    ),
    (
        "combined_embeddings_emb_025_ivfflat_idx",
        """
        CREATE INDEX IF NOT EXISTS combined_embeddings_emb_025_ivfflat_idx
            ON combined_embeddings USING ivfflat (emb_025 vector_cosine_ops)
            WITH (lists = 1215)
        """,
    ),
    (
        "combined_embeddings_emb_050_ivfflat_idx",
        """
        CREATE INDEX IF NOT EXISTS combined_embeddings_emb_050_ivfflat_idx
            ON combined_embeddings USING ivfflat (emb_050 vector_cosine_ops)
            WITH (lists = 1215)
        """,
    ),
    (
        "combined_embeddings_emb_075_ivfflat_idx",
        """
        CREATE INDEX IF NOT EXISTS combined_embeddings_emb_075_ivfflat_idx
            ON combined_embeddings USING ivfflat (emb_075 vector_cosine_ops)
            WITH (lists = 1215)
        """,
    ),
    (
        "combined_embeddings_emb_100_ivfflat_idx",
        """
        CREATE INDEX IF NOT EXISTS combined_embeddings_emb_100_ivfflat_idx
            ON combined_embeddings USING ivfflat (emb_100 vector_cosine_ops)
            WITH (lists = 1215)
        """,
    ),
    (
        "metadata_search_document_gin_idx",
        """
        CREATE INDEX IF NOT EXISTS metadata_search_document_gin_idx
            ON metadata USING gin (search_document)
        """,
    ),
    (
        "metadata_search_text_gin_idx",
        """
        CREATE INDEX IF NOT EXISTS metadata_search_text_gin_idx
            ON metadata USING gin (search_text gin_trgm_ops)
        """,
    ),
    (
        "metadata_mpd_occurrences_idx",
        """
        CREATE INDEX IF NOT EXISTS metadata_mpd_occurrences_idx
            ON metadata (mpd_occurrences DESC)
        """,
    ),
]

with connection.cursor() as cursor:
    cursor.execute("SET maintenance_work_mem = '4096MB'")
    cursor.execute("SET max_parallel_maintenance_workers = 7")
connection.commit()

for index_name, statement in indexes:
    with connection.cursor() as cursor:
        cursor.execute(statement)
    connection.commit()
    print(f"Created {index_name}")

print(f"Completion time: {time.time() - START_TIME} seconds")
connection.close()
