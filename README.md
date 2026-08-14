# Music recommendation algorithm visualisation

This project uses a combination of content based and collaborative based embeddings 
to determine where songs sit in 640 dimensional space. This space is used to fetch songs that are 
similar to a query song, by getting its nearest neighbours. The 640 dimensional space is also reduced down to 
3 dimensions, and songs plotted as points in 3D space to create a visual representation of how this algorithm would group songs

## Demo

### Example

Showing the nearest neighbours for the song "All I want for Christmas is you" shows that similar songs include other christmas songs and similar sounding songs.

<img width="1903" height="829" alt="screenshot" src="https://github.com/user-attachments/assets/c1d68a1f-efec-4436-b06e-73db11b1501b" />

### Video demonstration

https://github.com/user-attachments/assets/793eb4e9-4800-435c-940f-3f5698d23c80

## Tech stack

- Postgres / pgvector
- umap-learn
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
