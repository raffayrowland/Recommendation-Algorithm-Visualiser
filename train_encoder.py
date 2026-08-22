from itertools import islice
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

LYRIC_WEIGHT =     0.25
COLLAB_WEIGHT =    0.25
CLAP_WEIGHT =      0.25
ATTRIBUTE_WEIGHT = 0.25

BATCH_SIZE = 512
EPOCHS = 3

os.makedirs("models", exist_ok=True)

# Batch generator
def batches(items, batch_size=BATCH_SIZE):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


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


collab_data = np.load("embeddings/collab.npy", mmap_mode="r")
clap_data = np.load("embeddings/clap.npy", mmap_mode="r")
lyric_data = np.load("embeddings/lyrics.npy", mmap_mode="r")
attribute_data = np.load("embeddings/attributes.npy", mmap_mode="r")
split_data = np.load("embeddings/split.npy", mmap_mode="r")

train_indices = np.flatnonzero(split_data == 0)
val_indices = np.flatnonzero(split_data == 1)
test_indices = np.flatnonzero(split_data == 2)

model = MultimodalAutoencoder()
model.train()

optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.00001,
)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    batch_number = 0

    total_train_loss = 0
    total_train_collab = 0
    total_train_clap = 0
    total_train_lyric = 0
    total_train_attribute = 0
    total_train_examples = 0

    rng = np.random.default_rng(epoch)
    rng.shuffle(train_indices)

    for batch in batches(train_indices):
        # Prepare vectors
        collab_tensor = torch.from_numpy(np.array(collab_data[batch], copy=True))
        clap_tensor = torch.from_numpy(np.array(clap_data[batch], copy=True))
        lyric_tensor = torch.from_numpy(np.array(lyric_data[batch], copy=True))
        attribute_tensor = torch.from_numpy(np.array(attribute_data[batch], copy=True))

        optimiser.zero_grad(set_to_none=True)

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

        total_train_loss += total_loss.item() * current_batch_size
        total_train_collab += collab_loss.item() * current_batch_size
        total_train_clap += clap_loss.item() * current_batch_size
        total_train_lyric += lyric_loss.item() * current_batch_size
        total_train_attribute += attribute_loss.item() * current_batch_size

        total_train_examples += current_batch_size

        total_loss.backward()
        optimiser.step()

        if batch_number % 50 == 0:
            print(
                f"Batch {batch_number}: "
                f"total={total_loss.item():.4f}, "
                f"collab={collab_loss.item():.4f}, "
                f"clap={clap_loss.item():.4f}, "
                f"lyric={lyric_loss.item():.4f}, "
                f"attribute={attribute_loss.item():.4f}"
            )

        batch_number += 1

    avg_train_loss = total_train_loss / total_train_examples
    avg_train_collab = total_train_collab / total_train_examples
    avg_train_clap = total_train_clap / total_train_examples
    avg_train_lyric = total_train_lyric / total_train_examples
    avg_train_attribute = total_train_attribute / total_train_examples

    model.eval()
    total_val_loss = 0
    total_val_collab = 0
    total_val_clap = 0
    total_val_lyric = 0
    total_val_attribute = 0
    total_val_examples = 0

    with torch.no_grad():
        for batch in batches(val_indices):
            collab_tensor = torch.from_numpy(np.array(collab_data[batch], copy=True))
            clap_tensor = torch.from_numpy(np.array(clap_data[batch], copy=True))
            lyric_tensor = torch.from_numpy(np.array(lyric_data[batch], copy=True))
            attribute_tensor = torch.from_numpy(np.array(attribute_data[batch], copy=True))

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
            f"\nEPOCH: {epoch + 1}, \n"
            f"Train: total={avg_train_loss:.4f}, "
            f"collab={avg_train_collab:.4f}, "
            f"clap={avg_train_clap:.4f}, "
            f"lyric={avg_train_lyric:.4f}, "
            f"attribute={avg_train_attribute:.4f}\n"
            f"Val:   total={avg_val_loss:.4f}, "
            f"collab={avg_val_collab:.4f}, "
            f"clap={avg_val_clap:.4f}, "
            f"lyric={avg_val_lyric:.4f}, "
            f"attribute={avg_val_attribute:.4f}\n"
        )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "validation_loss": avg_val_loss,
            }, "models/linear_autoencoder_checkpoint.pt"
        )

    model.train()
