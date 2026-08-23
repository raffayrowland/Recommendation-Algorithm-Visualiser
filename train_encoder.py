from itertools import islice
import os
import numpy as np
import torch
import torch.nn.functional as F
from model import MultimodalAutoencoder

LYRIC_WEIGHT =     0.20
COLLAB_WEIGHT =    0.50
CLAP_WEIGHT =      0.25
ATTRIBUTE_WEIGHT = 0.05

BATCH_SIZE = 512
EPOCHS = 3
LOSSES = ["total", "collab", "clap", "lyric", "attribute", "examples"]  # Losses recorded during training

os.makedirs("models", exist_ok=True)

# Batch generator
def batches(items, batch_size=BATCH_SIZE):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def calculate_cosine_loss(reconstructed, target):
    similarity = F.cosine_similarity(reconstructed, target, dim=1)

    return 1 - similarity.mean()


# Takes in tensors and calculates the model's losses
def inference(clb_tensor, clp_tensor, lrc_tensor, att_tensor):
    (
        latent,
        collab_recon,
        clap_recon,
        lyric_recon,
        attribute_recon,
    ) = model(
        clb_tensor,
        clp_tensor,
        lrc_tensor,
        att_tensor,
    )

    clb_loss = calculate_cosine_loss(collab_recon, clb_tensor)
    clp_loss = calculate_cosine_loss(clap_recon, clp_tensor)
    lrc_loss = calculate_cosine_loss(lyric_recon, lrc_tensor)
    att_loss = calculate_cosine_loss(attribute_recon, att_tensor)

    ttl_loss = (
            COLLAB_WEIGHT * clb_loss +
            CLAP_WEIGHT * clp_loss +
            LYRIC_WEIGHT * lrc_loss +
            ATTRIBUTE_WEIGHT * att_loss
    )

    return  ttl_loss, clb_loss, clp_loss, lrc_loss, att_loss


# Stores the model's losses in a dictionary
def record_batch_losses(clb_tensor, clp_tensor, lrc_tensor, att_tensor, totals):
    losses = inference(clb_tensor, clp_tensor, lrc_tensor, att_tensor)
    batch_size = clb_tensor.shape[0]

    for name, loss in zip(["total", "collab", "clap", "lyric", "attribute"], losses):
        totals[name] += loss.item() * batch_size

    totals["examples"] += batch_size
    return losses


# Load the npy files containing the embeddings
collab_data = np.load("embeddings/collab.npy", mmap_mode="r")
clap_data = np.load("embeddings/clap.npy", mmap_mode="r")
lyric_data = np.load("embeddings/lyrics.npy", mmap_mode="r")
attribute_data = np.load("embeddings/attributes.npy", mmap_mode="r")
split_data = np.load("embeddings/split.npy", mmap_mode="r")

# Get the different splits
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
    train_loss = {name: 0.0 for name in LOSSES}

    rng = np.random.default_rng(epoch)
    rng.shuffle(train_indices)  # Shuffles the training examples each epoch

    for batch in batches(train_indices):
        # Prepare vectors
        collab_tensor = torch.from_numpy(np.array(collab_data[batch], copy=True))
        clap_tensor = torch.from_numpy(np.array(clap_data[batch], copy=True))
        lyric_tensor = torch.from_numpy(np.array(lyric_data[batch], copy=True))
        attribute_tensor = torch.from_numpy(np.array(attribute_data[batch], copy=True))

        optimiser.zero_grad(set_to_none=True)

        # Unpack losses for backpropogation
        total_loss, collab_loss, clap_loss, lyric_loss, attribute_loss = (
            record_batch_losses(
                collab_tensor, clap_tensor, lyric_tensor, attribute_tensor, train_loss
            )
        )

        # Do backpropogation
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

    avg_train_loss = train_loss["total"] / train_loss["examples"]
    avg_train_collab = train_loss["collab"] / train_loss["examples"]
    avg_train_clap = train_loss["clap"] / train_loss["examples"]
    avg_train_lyric = train_loss["lyric"] / train_loss["examples"]
    avg_train_attribute = train_loss["attribute"] / train_loss["examples"]

    model.eval()

    val_loss = {name: 0.0 for name in LOSSES}

    with torch.no_grad():
        for batch in batches(val_indices):
            collab_tensor = torch.from_numpy(np.array(collab_data[batch], copy=True))
            clap_tensor = torch.from_numpy(np.array(clap_data[batch], copy=True))
            lyric_tensor = torch.from_numpy(np.array(lyric_data[batch], copy=True))
            attribute_tensor = torch.from_numpy(np.array(attribute_data[batch], copy=True))

            # No need to unpack because no backpropogation
            record_batch_losses(collab_tensor, clap_tensor, lyric_tensor, attribute_tensor, val_loss)

        avg_val_loss = val_loss["total"] / val_loss["examples"]
        avg_val_collab = val_loss["collab"] / val_loss["examples"]
        avg_val_clap = val_loss["clap"] / val_loss["examples"]
        avg_val_lyric = val_loss["lyric"] / val_loss["examples"]
        avg_val_attribute = val_loss["attribute"] / val_loss["examples"]

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

    # Save model if its the best one yet
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "validation_loss": avg_val_loss,
            }, f"models/linear_CO{int(COLLAB_WEIGHT * 100)}"
               f"_CL{int(CLAP_WEIGHT * 100)}"
               f"_L{int(LYRIC_WEIGHT * 100)}"
               f"_A{int(ATTRIBUTE_WEIGHT * 100)}.pt"
        )

    model.train()
