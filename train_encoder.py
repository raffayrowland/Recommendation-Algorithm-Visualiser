from database import get_connection
from itertools import islice
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

LYRIC_WEIGHT =     0.25
COLLAB_WEIGHT =    0.25
CLAP_WEIGHT =      0.25
ATTRIBUTE_WEIGHT = 0.25

BATCH_SIZE = 128
EPOCHS = 3

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


def calculate_cosine_loss(reconstructed, target):
    similarity = F.cosine_similarity(reconstructed, target, dim=1)

    return 1 - similarity.mean()

# Multimodal autoencoder
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

    def forward(self, collab, clap, lyric, attribute):
        collab_proj = self.collab_projector(collab)
        clap_proj = self.clap_projector(clap)
        lyric_proj = self.lyric_projector(lyric)
        attribute_proj = self.attribute_projector(attribute)

        combined = torch.cat((collab_proj, clap_proj, lyric_proj, attribute_proj), dim=1)

        latent = self.fusion(combined)
        latent = F.normalize(latent, p=2, dim=1)

        collab_recon = self.collab_decoder(latent)
        clap_recon = self.clap_decoder(latent)
        lyric_recon = self.lyric_decoder(latent)
        attribute_recon = self.attribute_decoder(latent)

        return latent, collab_recon, clap_recon, lyric_recon, attribute_recon

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

model = MultimodalAutoencoder()
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.00001,
)

for epoch in range(EPOCHS):
    batch_number = 0
    random.shuffle(train)

    for batch in batches(train):
        # Prepare vectors
        embedding_batch = get_embedding_batch(batch)
        track_ids, collab_batch, clap_batch, lyric_batch, attribute_batch = prepare_batch(embedding_batch)

        # Convert to torch tensors
        collab_tensor = torch.from_numpy(collab_batch)
        clap_tensor = torch.from_numpy(clap_batch)
        lyric_tensor = torch.from_numpy(lyric_batch)
        attribute_tensor = torch.from_numpy(attribute_batch)

        optimizer.zero_grad(set_to_none=True)

        (
            latent,
            collab_recon,
            clap_recon,
            lyric_recon,
            attribute_recon,
        ) = model(
            collab_tensor,
            clap_tensor,
            lyric_tensor,
            attribute_tensor,
        )

        collab_loss = calculate_cosine_loss(collab_recon, collab_tensor)
        clap_loss = calculate_cosine_loss(clap_recon, clap_tensor)
        lyric_loss = calculate_cosine_loss(lyric_recon, lyric_tensor)
        attribute_loss = calculate_cosine_loss(attribute_recon, attribute_tensor)

        total_loss = (
            COLLAB_WEIGHT * collab_loss +
            CLAP_WEIGHT * clap_loss +
            LYRIC_WEIGHT * lyric_loss +
            ATTRIBUTE_WEIGHT * attribute_loss
        )

        total_loss.backward()
        optimizer.step()

        if batch_number % 10 == 0:
            print(
                f"Batch {batch_number}: "
                f"total={total_loss.item():.4f}, "
                f"collab={collab_loss.item():.4f}, "
                f"clap={clap_loss.item():.4f}, "
                f"lyric={lyric_loss.item():.4f}, "
                f"attribute={attribute_loss.item():.4f}"
            )

        batch_number += 1

    model.eval()
    total_val_loss = 0
    total_val_collab = 0
    total_val_clap = 0
    total_val_lyric = 0
    total_val_attribute = 0
    total_val_examples = 0

    with torch.no_grad():
        for batch in batches(val):
            val_embeddings = get_embedding_batch(batch)
            track_ids, collab_val, clap_val, lyric_val, attribute_val = prepare_batch(val_embeddings)

            collab_tensor = torch.from_numpy(collab_val)
            clap_tensor = torch.from_numpy(clap_val)
            lyric_tensor = torch.from_numpy(lyric_val)
            attribute_tensor = torch.from_numpy(attribute_val)

            (
                latent,
                collab_recon,
                clap_recon,
                lyric_recon,
                attribute_recon,
            ) = model(
                collab_tensor,
                clap_tensor,
                lyric_tensor,
                attribute_tensor,
            )

            collab_loss = calculate_cosine_loss(collab_recon, collab_tensor)
            clap_loss = calculate_cosine_loss(clap_recon, clap_tensor)
            lyric_loss = calculate_cosine_loss(lyric_recon, lyric_tensor)
            attribute_loss = calculate_cosine_loss(attribute_recon, attribute_tensor)

            total_loss = (
                    COLLAB_WEIGHT * collab_loss +
                    CLAP_WEIGHT * clap_loss +
                    LYRIC_WEIGHT * lyric_loss +
                    ATTRIBUTE_WEIGHT * attribute_loss
            )

            current_batch_size = collab_tensor.shape[0]

            total_val_loss += total_loss.item() * current_batch_size
            total_val_collab += collab_loss.item() * current_batch_size
            total_val_clap += clap_loss.item() * current_batch_size
            total_val_lyric += lyric_loss.item() * current_batch_size
            total_val_attribute += attribute_loss.item() * current_batch_size

            total_val_examples += current_batch_size

        avg_val_loss = total_val_loss / total_val_examples
        avg_val_collab = total_val_collab / total_val_examples
        avg_val_clap = total_val_clap / total_val_examples
        avg_val_lyric = total_val_lyric / total_val_examples
        avg_val_attribute = total_val_attribute / total_val_examples

        print(
            f"EPOCH: {epoch + 1}, "
            f"total={avg_val_loss:.4f}, "
            f"collab={avg_val_collab:.4f}, "
            f"clap={avg_val_clap:.4f}, "
            f"lyric={avg_val_lyric:.4f}, "
            f"attribute={avg_val_attribute:.4f}"
        )

    model.train()
