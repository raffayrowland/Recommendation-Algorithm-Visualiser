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


def add_metadata_bulk(rows):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO metadata (
                    track_id, 
                    isrc, 
                    track_name, 
                    artist_name, 
                    tag_list
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (track_id) DO NOTHING
                """, rows,
            )
        connection.commit()

    finally:
        connection.close()


def add_clap_embeddings_bulk(rows):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO clap_embeddings (track_id, embedding)
                SELECT %s, %s
                WHERE EXISTS (
                    SELECT 1
                    FROM metadata
                    WHERE metadata.track_id = %s
                )
                ON CONFLICT (track_id) DO NOTHING
                """,
                ((track_id, embedding, track_id) for track_id, embedding in rows),
            )
        connection.commit()

    finally:
        connection.close()


def add_cfbpr_bulk(rows):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO cf_bpr (track_id, embedding)
                SELECT %s, %s
                WHERE EXISTS (
                    SELECT 1
                    FROM metadata
                    WHERE metadata.track_id = %s
                )
                ON CONFLICT (track_id) DO NOTHING
                """,
                ((track_id, embedding, track_id) for track_id, embedding in rows),
            )
        connection.commit()

    finally:
        connection.close()


def get_random_songs(n):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT track_id, isrc, track_name FROM metadata ORDER BY RANDOM() LIMIT %s", (n,))
            return cursor.fetchall()
    finally:
        connection.close()


def get_cfbpr(track_id):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM cf_bpr WHERE track_id = %s", (track_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    finally:
        connection.close()


def get_clap(track_id):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM clap_embeddings WHERE track_id = %s", (track_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    finally:
        connection.close()
