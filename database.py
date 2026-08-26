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


def get_info_for_visualisation(n):
    # Get track_id, track_name, artist_name, isrc, combined_embedding for the top n songs
    with get_connection() as connection, connection.cursor() as cursor:
        results = cursor.execute("""
        SELECT
            t.track_id,
            t.track_name,
            t.artist_name,
            t.isrc,
            ce.embedding AS combined_embedding
        FROM tracks AS t
        JOIN combined_embedding AS ce
            ON ce.track_id = t.track_id
        ORDER BY t.mpd_occurrences DESC
        LIMIT %s
        """, (n,)).fetchall()

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
                SELECT %(query)s::TEXT AS text
            ),
            candidates AS MATERIALIZED (
                SELECT
                    t.track_id,
                    t.track_name,
                    t.artist_name,
                    t.search_text,
                    t.mpd_occurrences,
                    q.text,
                    word_similarity(q.text, t.search_text) AS word_score,
                    strict_word_similarity(q.text, t.search_text) AS strict_score
                FROM tracks AS t
                CROSS JOIN search_query AS q
                WHERE q.text <%% t.search_text
                ORDER BY
                    q.text <<-> t.search_text,
                    t.mpd_occurrences DESC,
                    t.track_id
                LIMIT GREATEST(%(limit)s * 10, 100)
            )
            SELECT
                track_id,
                track_name,
                artist_name
            FROM candidates
            ORDER BY
                starts_with(search_text, text) DESC,
                strict_score DESC,
                mpd_occurrences DESC,
                word_score DESC,
                track_id
            LIMIT %(limit)s
            """,
            {"query": query, "limit": limit},
        ).fetchall()

    return results


def get_nearest_neighbours(track_id, limit=50):
    query = sql.SQL(
        """
        WITH seed AS MATERIALIZED (
            SELECT embedding
            FROM combined_embedding
            WHERE track_id = %(track_id)s
        ),
        neighbours AS MATERIALIZED (
            SELECT
                candidate.track_id,
                candidate.embedding AS combined_embedding,
                candidate.embedding <=> seed.embedding AS distance
            FROM seed
            CROSS JOIN LATERAL (
                SELECT
                    indexed_candidate.track_id,
                    indexed_candidate.embedding
                FROM combined_embedding AS indexed_candidate
                ORDER BY indexed_candidate.embedding <=> seed.embedding
                LIMIT %(candidate_limit)s
            ) AS candidate
        )
        SELECT
            neighbours.track_id,
            tracks.track_name,
            tracks.artist_name,
            tracks.isrc,
            neighbours.combined_embedding
        FROM neighbours
        JOIN tracks USING (track_id)
        WHERE neighbours.track_id <> %(track_id)s
        ORDER BY neighbours.distance
        LIMIT %(limit)s
        """
    )
    with get_connection() as connection, connection.cursor() as cursor:
        parameters = {
            "track_id": track_id,
            "candidate_limit": limit + 1,
            "limit": limit,
        }
        cursor.execute("SET LOCAL hnsw.ef_search = 100")
        return cursor.execute(query, parameters).fetchall()


def get_info_single_song(track_id):
    with get_connection() as connection, connection.cursor() as cursor:
        return cursor.execute(
            """
            SELECT
                t.track_id,
                t.track_name,
                t.artist_name,
                t.isrc,
                e.embedding AS embedding
            FROM tracks AS t
            JOIN combined_embedding AS e USING (track_id)
            WHERE t.track_id = %(track_id)s
        """, {"track_id": track_id}).fetchone()
