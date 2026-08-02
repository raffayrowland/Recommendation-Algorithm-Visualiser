BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS combined_embeddings, clap_embeddings, cf_bpr, metadata;

CREATE TABLE metadata (
    track_id VARCHAR(22) PRIMARY KEY,
    isrc VARCHAR(12),
    track_name TEXT[] NOT NULL,
    artist_name TEXT[] NOT NULL,
    tag_list TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    mpd_occurrences INT DEFAULT 0,
    search_document TSVECTOR NOT NULL
);

CREATE TABLE clap_embeddings (
    track_id VARCHAR(22) PRIMARY KEY REFERENCES metadata,
    embedding vector(512) NOT NULL
);

CREATE TABLE cf_bpr (
    track_id VARCHAR(22) PRIMARY KEY REFERENCES metadata,
    embedding vector(128) NOT NULL
);

CREATE TABLE combined_embeddings (
    track_id VARCHAR(22) PRIMARY KEY REFERENCES metadata,
    emb_000 vector(640) NOT NULL,
    emb_025 vector(640) NOT NULL,
    emb_050 vector(640) NOT NULL,
    emb_075 vector(640) NOT NULL,
    emb_100 vector(640) NOT NULL
);

COMMIT;
