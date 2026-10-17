"""Inference-only Noise2Fret adapter.

The network layout is adapted from Riccardo Simionato and Louis Bigo's
MIT-licensed Noise2Fret implementation. Training-only losses, dataset loaders
and plotting code are intentionally excluded.
"""

from __future__ import annotations

from math import pi
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange, repeat


OPEN_PITCHES = (40, 45, 50, 55, 59, 64)
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE
EVENT_SLOTS = 2
STRING_COUNT = 6
FRET_CLASSES = 24
EMBEDDING_SIZE = 32


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("Number embedding dimension must be even")
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(dim + 1, features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flattened = rearrange(values, "... -> (...)")
        frequencies = (
            rearrange(flattened, "b -> b 1")
            * rearrange(self.weights, "d -> 1 d")
            * 2
            * pi
        )
        embedding = torch.cat(
            (
                rearrange(flattened, "b -> b 1"),
                frequencies.sin(),
                frequencies.cos(),
            ),
            dim=-1,
        )
        return self.to_out(embedding).view(*values.shape, self.features)


class Modulation(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(condition_dim, channels * 2)

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.proj(condition).chunk(2, dim=-1)
        return values * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        query, key, value = self.qkv(
            self.norm(values)
        ).view(values.shape[0], 3, values.shape[1], values.shape[2]).unbind(1)
        attention = torch.matmul(
            query.transpose(-2, -1),
            key,
        ) / (values.shape[1] ** 0.5)
        attention = self.dropout(functional.softmax(attention, dim=-1))
        return residual + self.dropout(
            self.proj(torch.matmul(value, attention.transpose(-2, -1)))
        )


class FeedForward(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.ff = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.ff(self.norm(values))


class InjectionBlock(nn.Module):
    def __init__(self, channels: int, injected_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(injected_channels, channels, 1)

    def forward(
        self,
        values: torch.Tensor,
        injected: torch.Tensor,
    ) -> torch.Tensor:
        if injected.shape[-1] != values.shape[-1]:
            injected = functional.interpolate(
                injected,
                size=values.shape[-1],
                mode="linear",
                align_corners=False,
            )
        return values + self.proj(injected)


class AudioEncoder(nn.Module):
    def __init__(self, output_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(32, output_channels, kernel_size=5, stride=4, padding=2),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encoder(audio.permute(0, 2, 1))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_dim: int,
        *,
        audio_channels: int,
        feature_channels: int,
        attention: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, input_channels),
            nn.SiLU(),
            nn.Conv1d(input_channels, output_channels, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, output_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv1d(output_channels, output_channels, 3, padding=1),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.modulation = Modulation(output_channels, time_dim)
        self.audio_injection = InjectionBlock(
            output_channels,
            audio_channels,
        )
        self.injection = InjectionBlock(
            output_channels,
            feature_channels,
        )
        self.attention = (
            SelfAttention(output_channels) if attention else None
        )
        self.feedforward = FeedForward(output_channels, dropout)

    def forward(
        self,
        values: torch.Tensor,
        time: torch.Tensor,
        audio: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.block1(values)
        hidden = hidden + self.time_proj(time).unsqueeze(-1)
        hidden = self.block2(hidden) + self.shortcut(values)
        hidden = self.modulation(hidden, time)
        hidden = self.injection(hidden, features)
        hidden = self.audio_injection(hidden, audio)
        if self.attention:
            hidden = self.attention(hidden)
        return self.feedforward(hidden)


class Noise2FretUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_channels = STRING_COUNT * EMBEDDING_SIZE
        base_channels = 64
        self.pos_emb = nn.Embedding(EVENT_SLOTS, input_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.input_dropout = nn.Dropout(0.1)
        self.input_conv = nn.Conv1d(
            input_channels,
            base_channels,
            3,
            padding=1,
        )
        self.c_conv = nn.Conv1d(515, base_channels, 3, padding=1)
        self.audio_encoder = AudioEncoder()
        block = lambda left, right, attention=False: ResidualBlock(
            left,
            right,
            128,
            audio_channels=64,
            feature_channels=base_channels,
            attention=attention,
        )
        self.down1 = block(64, 128)
        self.down2 = block(128, 256)
        self.down3 = block(256, 512)
        self.down4 = block(512, 512)
        self.downsample = nn.Conv1d(512, 512, 3, stride=2, padding=1)
        self.mid = block(512, 512, attention=True)
        self.upsample = nn.ConvTranspose1d(512, 512, 4, stride=2, padding=1)
        self.up1 = block(1024, 256)
        self.up2 = block(768, 128)
        self.up3 = block(384, 64)
        self.up4 = block(192, 64)
        self.output = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, input_channels, 3, padding=1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        time: torch.Tensor,
        audio: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        positions = self.pos_emb(
            torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        ).permute(0, 2, 1)
        values = self.input_dropout(
            self.input_conv(tokens.permute(0, 2, 1) + positions)
        )
        time = self.time_emb(time)
        features = self.c_conv(features.permute(0, 2, 1))
        audio = self.audio_encoder(audio)
        skip1 = self.down1(values, time, audio, features)
        skip2 = self.down2(skip1, time, audio, features)
        skip3 = self.down3(skip2, time, audio, features)
        skip4 = self.down4(skip3, time, audio, features)
        values = self.mid(self.downsample(skip4), time, audio, features)
        values = self.upsample(values)
        if values.shape[-1] != skip4.shape[-1]:
            values = functional.interpolate(
                values,
                size=skip4.shape[-1],
                mode="linear",
                align_corners=False,
            )
        values = self.up1(torch.cat((values, skip4), dim=1), time, audio, features)
        values = self.up2(torch.cat((values, skip3), dim=1), time, audio, features)
        values = self.up3(torch.cat((values, skip2), dim=1), time, audio, features)
        values = self.up4(torch.cat((values, skip1), dim=1), time, audio, features)
        return self.output(values).permute(0, 2, 1)


class Noise2FretRuntime(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.model = Noise2FretUNet().to(device)
        self.time_encoder = NumberEmbedder(128).to(device)
        self.embeddings = nn.Embedding(FRET_CLASSES, EMBEDDING_SIZE).to(device)

    def load(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.time_encoder.load_state_dict(
            checkpoint["time_embedding_state_dict"]
        )
        self.embeddings.load_state_dict(checkpoint["embedding_state_dict"])
        self.eval()

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        reshaped = values.view(
            values.shape[0],
            values.shape[1],
            STRING_COUNT,
            EMBEDDING_SIZE,
        )
        return reshaped @ self.embeddings.weight.T

    @torch.inference_mode()
    def predict(
        self,
        audio: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        seed_ids = torch.zeros(
            (audio.shape[0], EVENT_SLOTS, STRING_COUNT),
            dtype=torch.long,
            device=self.device,
        )
        embedded = self.embeddings(seed_ids).view(
            audio.shape[0],
            EVENT_SLOTS,
            STRING_COUNT * EMBEDDING_SIZE,
        )
        torch.manual_seed(0)
        noisy = torch.randn_like(embedded)
        sigmas = torch.linspace(1.0, 0.0, 2, device=self.device)
        sigma_batch = repeat(sigmas, "i -> i b 1 1", b=audio.shape[0])
        alpha = torch.cos(sigma_batch * pi / 2)
        beta = torch.sin(sigma_batch * pi / 2)
        time = repeat(
            self.time_encoder(sigmas),
            "i d -> i b d",
            b=audio.shape[0],
        )
        velocity = self.model(noisy, time[0], audio, features)
        clean = alpha[0] * noisy - beta[0] * velocity
        noise = beta[0] * noisy + alpha[0] * velocity
        denoised = alpha[1] * clean + beta[1] * noise
        return self.decode(denoised).argmax(dim=-1)


def _device() -> torch.device:
    configured = os.environ.get("NOISE2FRET_DEVICE")
    if configured:
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _audio_features(audio: torch.Tensor) -> torch.Tensor:
    mono = audio.squeeze(-1)
    window = torch.hann_window(1024, device=mono.device)
    magnitude = torch.stft(
        mono,
        n_fft=1024,
        hop_length=256,
        window=window,
        return_complex=True,
    ).abs()
    flux = functional.pad(
        torch.diff(magnitude, dim=-1).clamp(min=0).sum(dim=1),
        (1, 0),
    )
    frequencies = torch.linspace(
        0,
        SAMPLE_RATE / 2,
        magnitude.shape[1],
        device=mono.device,
    )
    total = magnitude.square().sum(dim=1)
    bright = magnitude[:, frequencies >= 1500, :].square().sum(dim=1)
    brightness = bright / total.clamp(min=1e-8)
    return torch.cat(
        (
            rearrange(magnitude, "b f t -> b t f"),
            flux.unsqueeze(-1),
            brightness.unsqueeze(-1),
        ),
        dim=-1,
    )


def predict_position_priors(
    audio_path: Path,
    checkpoint_path: Path,
    window_indices: list[int],
    batch_size: int = 8,
) -> dict[int, list[list[tuple[int, int, int]]]]:
    if not window_indices:
        return {}
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )
    device = _device()
    runtime = Noise2FretRuntime(device)
    runtime.load(checkpoint_path)
    predictions: dict[int, list[list[tuple[int, int, int]]]] = {}
    for start in range(0, len(window_indices), batch_size):
        indices = window_indices[start : start + batch_size]
        windows = []
        for index in indices:
            segment = audio[
                index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES
            ]
            windows.append(
                np.pad(
                    segment,
                    (0, max(0, WINDOW_SAMPLES - len(segment))),
                )[:WINDOW_SAMPLES]
            )
        tensor = torch.from_numpy(np.stack(windows)).float().unsqueeze(-1)
        tensor = tensor.to(device)
        classes = runtime.predict(tensor, _audio_features(tensor)).cpu()
        for window_index, window_classes in zip(indices, classes, strict=True):
            slots: list[list[tuple[int, int, int]]] = []
            for slot in window_classes:
                positions = []
                for source_string, value in enumerate(slot.tolist()):
                    if value <= 0:
                        continue
                    fret = value - 1
                    positions.append(
                        (
                            STRING_COUNT - source_string,
                            fret,
                            OPEN_PITCHES[source_string] + fret,
                        )
                    )
                slots.append(positions)
            predictions[window_index] = slots
    return predictions
