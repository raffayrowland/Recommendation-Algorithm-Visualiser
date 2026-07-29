import psycopg
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

    return connection

def add_metadata(track_id, isrc, track_name, artist_name, tag_list):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
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
                """, (track_id, isrc, track_name, artist_name, tag_list)
            )
            connection.commit()

    finally:
        connection.close()