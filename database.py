import os
import unicodedata
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

def normalise_search_text(value):
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    characters = []

    for character in decomposed:
        if unicodedata.combining(character):
            continue
        characters.append(character if character.isalnum() else " ")

    return " ".join("".join(characters).split())


def get_connection():
    connection = psycopg.connect(
        host="localhost",
        port=5432,
        user=os.getenv("DB_USER"),
        dbname=os.getenv("DB_NAME"),
        password=os.getenv("DB_PASSWORD"),
    )
    register_vector(connection)

    return connection


def get_info_for_visualisation(n, alpha):
    # Get track_id, track_name, artist_name, isrc, combined_embedding for the top n songs
    with get_connection() as connection, connection.cursor() as cursor:
        sql = f"""
        SELECT
            m.track_id,
            m.track_name,
            m.artist_name,
            m.isrc,
            ce.emb_{alpha} AS combined_embedding
        FROM metadata AS m
        JOIN combined_embeddings AS ce
            ON ce.track_id = m.track_id
        ORDER BY m.mpd_occurrences DESC
        LIMIT %s
        """
        cursor.execute(sql, (n,))
        results = cursor.fetchall()

    return results

def search_for_song_by_name(query, n=5):
    query = normalise_search_text(query)
    if len(query) < 2:
        return []

    limit = min(max(int(n), 1), 50)  # Get a max of n results if n < 50, else 50
    with get_connection() as connection, connection.cursor() as cursor:
        results = cursor.execute(
            """
            WITH search_query AS (
                SELECT
                    %(query)s::TEXT AS text,
                    plainto_tsquery('simple', %(query)s) AS document
            )
            SELECT
                m.track_id,
                m.track_name,
                m.artist_name
            FROM metadata AS m
            CROSS JOIN search_query AS q
            WHERE
                m.search_document @@ q.document
                OR m.search_text %% q.text
                OR q.text <%% m.search_text
            ORDER BY
                CASE
                    WHEN m.search_document @@ q.document
                        THEN ts_rank_cd(m.search_document, q.document)
                    ELSE 0
                END DESC,
                strict_word_similarity(q.text, m.search_text) DESC,
                m.mpd_occurrences DESC,
                m.track_id
            LIMIT %(limit)s
            """,
            {"query": query, "limit": limit},
        ).fetchall()

    return results


def get_nearest_neighbours(track_id, alpha, limit=50):
    embedding_column = sql.Identifier(f"emb_{alpha}")
    query = sql.SQL(
        """
        WITH seed AS MATERIALIZED (
            SELECT {embedding} AS embedding
            FROM combined_embeddings
            WHERE track_id = %(track_id)s
        ),
        neighbours AS MATERIALIZED (
            SELECT
                candidate.track_id,
                candidate.{embedding} AS combined_embedding,
                candidate.{embedding} <=> (SELECT embedding FROM seed) AS distance
            FROM combined_embeddings AS candidate
            WHERE EXISTS (SELECT 1 FROM seed)
            ORDER BY distance
            LIMIT %(candidate_limit)s
        )
        SELECT
            neighbours.track_id,
            metadata.track_name,
            metadata.artist_name,
            metadata.isrc,
            neighbours.combined_embedding
        FROM neighbours
        JOIN metadata USING (track_id)
        WHERE neighbours.track_id <> %(track_id)s
        ORDER BY neighbours.distance
        LIMIT %(limit)s
        """
    ).format(embedding=embedding_column)

    parameters = {
        "track_id": track_id,
        "candidate_limit": limit + 1,
        "limit": limit,
    }

    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL ivfflat.probes = 35")
        return cursor.execute(query, parameters).fetchall()
