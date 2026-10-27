"""
UWrites - Neural Network Architectures for Personalized Handwriting Generation
Implements BiGRU TextEncoder, DCGAN Generator with Transposed Convolutions,
and Discriminator modules.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Bidirectional GRU text encoder that maps a sequence of character token IDs
    into a dense 128-dimensional conditioning vector.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=True)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            Conditioning vector of shape (batch_size, 2 * embed_dim)
        """
        e = self.embed(text_ids)
        _, h = self.gru(e)
        # Concatenate forward h[0] and backward h[1] hidden states
        return torch.cat([h[0], h[1]], dim=1)


class Generator(nn.Module):
    """
    Deep Convolutional GAN Generator that synthesizes a (1, 64, 256) grayscale handwriting image
    conditioned on a Gaussian latent noise vector and a text conditioning feature vector.
    """
    def __init__(self, z_dim: int = 128, text_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(z_dim + text_dim, 512 * 4 * 16)
        self.net = nn.Sequential(
            # Spatial dimension: 4 x 16 -> 8 x 32
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # 8 x 32 -> 16 x 64
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # 16 x 64 -> 32 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # 32 x 128 -> 64 x 256
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent noise vector (batch_size, z_dim)
            text_feat: Condition text vector (batch_size, text_dim)
        Returns:
            Generated image tensor of shape (batch_size, 1, 64, 256) in [-1.0, 1.0]
        """
        x = torch.cat([z, text_feat], dim=1)
        x = self.fc(x).view(-1, 512, 4, 16)
        return self.net(x)


class Discriminator(nn.Module):
    """
    Convolutional Discriminator determining whether an input handwriting image is real or synthetic.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (1, 64, 256) -> (64, 32, 128)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 32, 128) -> (128, 16, 64)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 16, 64) -> (256, 8, 32)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 8, 32) -> (512, 4, 16)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # (512, 4, 16) -> (1, 1, 13)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor of shape (batch_size, 1, 64, 256)
        Returns:
            Logit scores of shape (batch_size,)
        """
        return self.net(img).view(img.size(0), -1).mean(dim=1)


class ConditionalDiscriminator(nn.Module):
    """
    Enhanced Conditional Discriminator using projection conditioning.
    Pairs visual feature maps with text embeddings to ensure generated handwriting
    matches the requested lexical transcription.
    """
    def __init__(self, text_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear_out = nn.Linear(512, 1)
        self.embed_proj = nn.Linear(text_dim, 512)

    def forward(self, img: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        feat = self.features(img)
        # Global average pool over spatial dimensions (H, W) -> (batch_size, 512)
        feat_pooled = torch.mean(feat, dim=(2, 3))
        
        # Unconditional real/fake score
        out = self.linear_out(feat_pooled)
        
        # Conditional projection inner product
        proj = self.embed_proj(text_feat)
        cond_score = torch.sum(feat_pooled * proj, dim=1, keepdim=True)
        
        return (out + cond_score).view(-1)
