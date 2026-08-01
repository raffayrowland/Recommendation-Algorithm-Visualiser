import psycopg
from pgvector.psycopg import register_vector
import os
from dotenv import load_dotenv

load_dotenv()

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


def search_for_song_by_name(query):
    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT 
                track_id, track_name, artist_name
            FROM metadata
            WHERE search_document @@ websearch_to_tsquery('simple', %s)
            LIMIT 5
            """, (query,)
        )
        results = cursor.fetchall()

    return results


def search_for_song_by_id(track_id):
    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT track_id, track_name, artist_name
            FROM metadata
            WHERE track_id = %s
            """, (track_id,)
        )
        result = cursor.fetchone()

    return result


def get_nearest_neighbours(track_id, alpha):
    connection = get_connection()
    match alpha:
        case 0:
            alpha = "000"
        case 0.25:
            alpha = "025"
        case 0.5:
            alpha = "050"
        case 0.75:
            alpha = "075"
        case 1:
            alpha = "100"

    with connection.cursor() as cursor:
        sql = f"""
            SELECT emb_{alpha}
            FROM combined_embeddings
            WHERE track_id = %s;
            """
        query_embedding = cursor.execute(sql, (track_id,)).fetchone()

        if query_embedding is None:
            return []

        sql = f"""
            SELECT
                track_id,
                emb_{alpha} <=> %(embedding)s AS cosine_distance
            FROM combined_embeddings
            ORDER BY emb_{alpha} <=> %(embedding)s
            LIMIT 11;
            """
        results = cursor.execute(sql, {"embedding": query_embedding[0]},).fetchall()

    # The queried track will normally be the closest result.
    return [
        result
        for result in results
        if result[0] != track_id
    ][:10]
