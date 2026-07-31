BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE metadata (
    track_id VARCHAR(22) PRIMARY KEY,
    isrc VARCHAR(12) NOT NULL,
    track_name TEXT NOT NULL,
    artist_name TEXT NOT NULL,
    tag_list TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],

    CONSTRAINT metadata_track_id_length
        CHECK (char_length(track_id) = 22),

    CONSTRAINT metadata_isrc_length
        CHECK (isrc IS NULL OR char_length(isrc) = 12)
);

CREATE TABLE clap_embeddings (
    track_id VARCHAR(22) PRIMARY KEY,
    embedding vector(512) NOT NULL,

    CONSTRAINT clap_embeddings_track_fk
        FOREIGN KEY (track_id)
        REFERENCES metadata(track_id)
        ON DELETE CASCADE
);

CREATE TABLE cf_bpr (
    track_id VARCHAR(22) PRIMARY KEY,
    embedding vector(128) NOT NULL,

    CONSTRAINT cf_bpr_track_fk
        FOREIGN KEY (track_id)
        REFERENCES metadata(track_id)
        ON DELETE CASCADE
);

CREATE TABLE combined_embeddings (
    track_id VARCHAR(22) PRIMARY KEY,
    emb_025 vector(640) NOT NULL,
    emb_050 vector(640) NOT NULL,
    emb_075 vector(640) NOT NULL,
    emb_100 vector(640) NOT NULL,

    CONSTRAINT combined_track_fk
        FOREIGN KEY (track_id)
        REFERENCES metadata(track_id)
        ON DELETE CASCADE
);

CREATE INDEX metadata_isrc_index
    ON metadata(isrc);

CREATE INDEX metadata_tag_list_index
    ON metadata
    USING GIN(tag_list);

COMMIT;