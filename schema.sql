CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TABLE IF EXISTS tracks, track_embeddings, combined_embedding;

CREATE TABLE tracks (
    track_id VARCHAR(22) PRIMARY KEY,
    isrc VARCHAR(12),
    track_name TEXT NOT NULL,
    artist_names TEXT[] NOT NULL,
    mpd_occurrences INT DEFAULT 0,
    search_text TEXT NOT NULL
);

CREATE TABLE track_embeddings (
    track_id VARCHAR(22) PRIMARY KEY,
    collab vector(128),
    clap vector(512),
    lyric vector(1024),
    attributes vector(1024)
);

CREATE TABLE combined_embedding (
    track_id VARCHAR(22) PRIMARY KEY,
    embedding vector(256)
);
