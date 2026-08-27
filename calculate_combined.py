import torch
import os
from model import MultimodalAutoencoder

LYRIC_WEIGHT =     0.20
COLLAB_WEIGHT =    0.45
CLAP_WEIGHT =      0.25
ATTRIBUTE_WEIGHT = 0.10
MODEL_NAME = (f"models/linear_CO{int(COLLAB_WEIGHT * 100)}"
              f"_CL{int(CLAP_WEIGHT * 100)}"
              f"_L{int(LYRIC_WEIGHT * 100)}"
              f"_A{int(ATTRIBUTE_WEIGHT * 100)}.pt")

device ='cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(MODEL_NAME):
    raise ValueError(f"Model {MODEL_NAME} does not exist. You must train one first")

model = MultimodalAutoencoder().to(device)
checkpoint = torch.load(MODEL_NAME, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def get_embeddings(collab, clap, lyrics, attributes):
    collab = torch.as_tensor(collab, dtype=torch.float32, device=device)
    clap = torch.as_tensor(clap, dtype=torch.float32, device=device)
    lyrics = torch.as_tensor(lyrics, dtype=torch.float32, device=device)
    attributes = torch.as_tensor(attributes, dtype=torch.float32, device=device)

    with torch.inference_mode():
        combined, _, _, _, _ = model(collab, clap, lyrics, attributes)

    return combined.cpu().numpy()