# Music recommendation algorithm visualisation

This project computes a 256 dimensional vector representation of 1.4 million songs using a weighted combination of each song's lyrics, 
structure, attributes (key, tempo etc.), and collaborative filtering vector. This 'combined vector' is used to find similar songs to 
a query song using nearest neighbour search. The 256 dimensional embedding space is also reduced to 3 dimensions to provide 
an intuitive visualisation of how the algorithm groups songs. 

## Approach

The dataset used for this project (talkplay-data-extra) includes a 128 dimensional collaborative filtering vector, 512 
dimensional CLAP embedding, and two 1024 dimensional qwen-0.6b embeddings, one for lyrics and one for attributes. This
project uses a linear autoencoder to reduce all of these vectors down to a single 256 dimensional embedding, which is used 
for nearest neighbour search. This has a few benefits: 

- Similar ideas from different modalities are combined into one dimension, rather than spread across many. For example, 
the meaning "Christmas" may be represented in lyrics, structure, and collaborative filtering separately. The autoencoder 
can learn to combine these meanings into a single dimension, improving similarity search accuracy.
- Nearest neighbour search on a 256 dimensional embedding space is much quicker than on a higher dimensional space. 
- Storing and indexing a 256 dimensional embedding space takes up much less storage than a higher dimensional space.

The autoencoder is trained alongside a decoder. The encoder produces the 256 dimensional embedding, 
and the decoder attempts to reconstruct the original collaborative, CLAP, lyric, and attribute embeddings
from the combined embedding, using cosine similarity as a loss metric. The decoder is only used for training
purposes.

## Demo

### Example

Showing the nearest neighbours for the song "All I want for Christmas is you" shows that similar songs include other christmas songs and similar sounding songs.

<img width="1913" height="826" alt="Screenshot from 2026-08-30 01-27-09" src="https://github.com/user-attachments/assets/5b463ca7-2455-41e0-9139-0096609f512e" />

### Video demonstration

[3D Space view](https://github.com/user-attachments/assets/eb5a5c7e-b53b-4de1-95c6-83902d634fd6)

[Nearest neighbours for some songs](https://github.com/user-attachments/assets/9856d965-e959-4bfb-8176-56c6dbd91a3c)

## Prerequisites

The setup below is tested with Ubuntu 24.04, Python 3.12, and PostgreSQL 18. The preprocessed database dump was created with
PostgreSQL 18

- Python 3.12
- PostgreSQL 18
- pgvector for PostgreSQL 18
- curl

If you need to install any of these, use one of these commands

```
# Install curl
sudo apt install curl

# Install python3.12
sudo apt install python3.12

# Install postgreSQL 18
sudo apt install -y postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
sudo apt install -y postgresql-18

# Install pgvector
sudo apt install -y postgresql-18-pgvector
```

The preprocessed database dump was created with PostgreSQL 18, so using the same major version is recommended.

## Installation

Clone the repo and install dependencies
```
git clone https://github.com/raffayrowland/Recommendation-Algorithm-Visualiser.git
cd Recommendation-Algorithm-Visualiser
curl -L "https://drive.usercontent.google.com/download?id=1k7XxxjO4aWGqhGM1ebcqjBCCxhQ3MnC9&export=download&confirm=t" -o database.dump
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Create the database and import from the dump. Ensure PostgreSQL 18 is installed

```
sudo -u postgres psql
```

At the postgres=# prompt, run
```
CREATE ROLE music_recommender_owner
    WITH LOGIN
    PASSWORD 'password';

CREATE DATABASE music_recommender
    OWNER music_recommender_owner;

\connect music_recommender
```
```
CREATE EXTENSION vector;

SET ROLE music_recommender_owner;
CREATE EXTENSION pg_trgm;
RESET ROLE;

\dx
\quit
```

Make the .env file (note: if you changed the database name, owner, or password, you must also change them here)
```
touch .env
cat >> .env << EOF
DB_USER=music_recommender_owner
DB_NAME=music_recommender
DB_PASSWORD=password
EOF
```

Restore the database from the terminal containing the dump file. Adjust the maintenance_work_mem and parallel workers
to fit your machine

```
sudo -u postgres /usr/lib/postgresql/18/bin/psql \
  --port=5432 \
  --dbname=music_recommender \
  --command='CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;'
PGOPTIONS="-c maintenance_work_mem=6GB \
-c max_parallel_maintenance_workers=7 \
-c max_parallel_workers=8" \
pg_restore \
  --host=localhost \
  --port=5432 \
  --username=music_recommender_owner \
  --password \
  --dbname=music_recommender \
  --exit-on-error \
  --verbose \
  database.dump
```

Start the application

```
source venv/bin/activate
python server.py
```

## Tech stack

- Postgres / pgvector
- umap-learn
- torch
- FastAPI
- pandas
- Deezer API

## Data

This project uses talkplay-data-extra, which contains data for over 1.4 million songs:

[TalkPlay Data Extra](https://huggingface.co/collections/talkpl-ai/talkplay-data-extra)

## Citations and attribution

The main external data, methods, and media services used by this project are:

- **TalkPlay Data Extra.** Track metadata and the precomputed CF-BPR and LAION-CLAP embeddings come from the [TalkPlay Data Extra collection](https://huggingface.co/collections/talkpl-ai/talkplay-data-extra).
- **Spotify Million Playlist Dataset.** Playlist occurrence counts used to order tracks come from the [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). Cite: C.-W. Chen, P. Lamere, M. Schedl, and H. Zamani, "RecSys Challenge 2018: Automatic Music Playlist Continuation," *Proceedings of the 12th ACM Conference on Recommender Systems*, 2018. [doi:10.1145/3240323.3240342](https://doi.org/10.1145/3240323.3240342).
- **LAION-CLAP.** Y. Wu et al., "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," *ICASSP*, 2023. [doi:10.1109/ICASSP49357.2023.10095969](https://doi.org/10.1109/ICASSP49357.2023.10095969).
- **Bayesian Personalized Ranking.** S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback," *Proceedings of UAI*, 2009, pp. 452-461. [Paper](https://arxiv.org/abs/1205.2618).
- **UMAP.** L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," 2018. [Paper](https://arxiv.org/abs/1802.03426).
- **Deezer API.** Audio previews and album artwork are fetched at runtime from the [Deezer API](https://developers.deezer.com/api); those media remain the property of their respective rights holders.

The Spotify dataset is limited to non-commercial research use and must not be redistributed; see its [terms](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/challenge_rules). The TalkPlay Data Extra pages do not currently state a licence, so permission should be confirmed before redistributing that data. The datasets and Deezer media are not included in this repository.
