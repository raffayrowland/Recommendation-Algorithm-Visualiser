from torch import nn
import torch
import torch.nn.functional as F


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