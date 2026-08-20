from database import get_connection
from itertools import islice
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

LYRIC_WEIGHT =     0.15
COLLAB_WEIGHT =    0.5
CLAP_WEIGHT =      0.25
ATTRIBUTE_WEIGHT = 0.1

BATCH_SIZE = 512

connection = get_connection()

# Batch generator
def batches(items, batch_size=BATCH_SIZE):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch

# Gets the embeddings for a batch of track_ids
def get_embedding_batch(ids):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT track_id, collab, clap, lyric, attributes
            FROM track_embeddings
            WHERE track_id = ANY(%s)
            """, (ids,)
        )

        return cursor.fetchall()


# Prepares the embeddings for a batch into L2 normalised batch matrices
def prepare_batch(rows):
    track_ids = []
    collab_vectors = []
    clap_vectors = []
    lyric_vectors = []
    attribute_vectors = []

    # Convert the embeddings to numpy format and store in a list
    for track_id, collab, clap, lyric, attribute in rows:
        track_ids.append(track_id)
        collab_vectors.append(np.asarray(collab.to_numpy(), dtype=np.float32))
        clap_vectors.append(np.asarray(clap.to_numpy(), dtype=np.float32))
        lyric_vectors.append(np.asarray(lyric.to_numpy(), dtype=np.float32))
        attribute_vectors.append(np.asarray(attribute.to_numpy(), dtype=np.float32))

    # Stack them into 2D matrices
    collab_batch = np.stack(collab_vectors, axis=0)
    clap_batch = np.stack(clap_vectors, axis=0)
    lyric_batch = np.stack(lyric_vectors, axis=0)
    attribute_batch = np.stack(attribute_vectors, axis=0)

    # L2 Normalisation
    collab_batch = collab_batch / np.linalg.norm(collab_batch, axis=1, keepdims=True)
    clap_batch = clap_batch / np.linalg.norm(clap_batch, axis=1, keepdims=True)
    lyric_batch = lyric_batch / np.linalg.norm(lyric_batch, axis=1, keepdims=True)
    attribute_batch = attribute_batch / np.linalg.norm(attribute_batch, axis=1, keepdims=True)

    return track_ids, collab_batch, clap_batch, lyric_batch, attribute_batch


# Encoder
class MultimodalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Projector layers
        self.collab_projector = nn.Linear(128, 128)
        self.clap_projector = nn.Linear(512, 128)
        self.lyric_projector = nn.Linear(1024, 128)
        self.attribute_projector = nn.Linear(1024, 128)

        # Fusion layer
        self.fusion = nn.Linear(512, 256)

        # Decoders
        self.collab_decoder = nn.Linear(256, 128)
        self.clap_decoder = nn.Linear(256, 512)
        self.lyric_decoder = nn.Linear(256, 1024)
        self.attribute_decoder = nn.Linear(256, 1024)

# Get a sample of 100,000 random track_ids
with connection.cursor() as cursor:
    cursor.execute(
        """
        SELECT track_id
        FROM track_embeddings
        ORDER BY random()
        LIMIT 100000
        """
    )
    track_id_rows = cursor.fetchall()
    track_ids = [row[0] for row in track_id_rows]

# Split into train, test, and validation splits
samples = len(track_ids)
train = track_ids[:int(samples * 0.9)]
test = track_ids[int(samples * 0.9):int(samples * 0.95)]
val = track_ids[int(samples * 0.95):]

for batch in batches(train):
    embedding_batch = get_embedding_batch(batch)
    track_ids, collab_batch, clap_batch, lyric_batch, attribute_batch = prepare_batch(embedding_batch)

    collab_tensor = torch.from_numpy(collab_batch)
    clap_tensor = torch.from_numpy(clap_batch)
    lyric_tensor = torch.from_numpy(lyric_batch)
    attribute_tensor = torch.from_numpy(attribute_batch)


