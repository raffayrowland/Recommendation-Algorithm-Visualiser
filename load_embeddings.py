from database import get_connection
import numpy as np
import hashlib
import os

os.makedirs("embeddings", exist_ok=True)
CHUNK_SIZE = 50_000


def main():
    with get_connection() as connection, connection.cursor() as cursor:
        row_count = cursor.execute("""
        SELECT COUNT(*) FROM track_embeddings
        """).fetchone()[0]

    # Create files and define shape
    collab_file = np.lib.format.open_memmap(
        "embeddings/collab.npy",
        mode="w+",
        dtype=np.float32,
        shape=(row_count, 128)
    )

    clap_file = np.lib.format.open_memmap(
        "embeddings/clap.npy",
        mode="w+",
        dtype=np.float32,
        shape=(row_count, 512)
    )

    lyrics_file = np.lib.format.open_memmap(
        "embeddings/lyrics.npy",
        mode="w+",
        dtype=np.float32,
        shape=(row_count, 1024)
    )

    attributes_file = np.lib.format.open_memmap(
        "embeddings/attributes.npy",
        mode="w+",
        dtype=np.float32,
        shape=(row_count, 1024)
    )

    split_file = np.lib.format.open_memmap(
        "embeddings/split.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(row_count,),
    )

    track_id_file = np.lib.format.open_memmap(
        "embeddings/track_id.npy",
        mode="w+",
        dtype=np.dtype("<U22"),
        shape=(row_count,),
    )

    connection = get_connection()
    cursor = connection.cursor(name="embedding_export_serverside")
    cursor.execute(
        """
        SELECT track_id, collab, clap, lyric, attributes
        FROM track_embeddings
        """, binary=True)

    rows = cursor.fetchmany(CHUNK_SIZE)

    rows_written = 0
    while rows:
        track_id_chunk = np.asarray(
            [row[0] for row in rows],
            dtype=np.dtype("<U22"),
        )

        collab_chunk = np.stack(
            [row[1].to_numpy() for row in rows]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(collab_chunk, axis=1, keepdims=True)
        collab_chunk /= np.maximum(norms, 1e-12)

        clap_chunk = np.stack(
            [row[2].to_numpy() for row in rows]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(clap_chunk, axis=1, keepdims=True)
        clap_chunk /= np.maximum(norms, 1e-12)

        lyric_chunk = np.stack(
            [row[3].to_numpy() for row in rows]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(lyric_chunk, axis=1, keepdims=True)
        lyric_chunk /= np.maximum(norms, 1e-12)

        attributes_chunk = np.stack(
            [row[4].to_numpy() for row in rows]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(attributes_chunk, axis=1, keepdims=True)
        attributes_chunk /= np.maximum(norms, 1e-12)

        split_chunk = np.empty(len(rows), dtype=np.uint8)

        start = rows_written
        stop = start + len(rows)

        for index, row in enumerate(rows):
            track_id = row[0]

            digest = hashlib.md5(track_id.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little") % 100

            if bucket < 95:
                split_chunk[index] = 0  # Training
            else:
                split_chunk[index] = 1  # Validation

        track_id_file[start:stop] = track_id_chunk
        collab_file[start:stop] = collab_chunk
        clap_file[start:stop] = clap_chunk
        lyrics_file[start:stop] = lyric_chunk
        attributes_file[start:stop] = attributes_chunk
        split_file[start:stop] = split_chunk

        rows_written = stop
        rows = cursor.fetchmany(CHUNK_SIZE)

        print(f"\r{rows_written:,}/{row_count:,} rows exported", end="")

    print()
    track_id_file.flush()
    collab_file.flush()
    clap_file.flush()
    lyrics_file.flush()
    attributes_file.flush()
    split_file.flush()

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()