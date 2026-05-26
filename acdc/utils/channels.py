"""Shared helpers for multi-channel image lists."""

from __future__ import annotations

from collections.abc import Sequence

from acdc.core.data import AcdcData
from acdc.utils.display_levels import stack_display_levels


def default_channel_weights(channel_count: int) -> list[float]:
    return [1.0] * channel_count


def resize_channel_weights(weights: Sequence[float], channel_count: int) -> list[float]:
    """Keep existing weights when channels change; pad or trim with 1.0."""
    current = [max(0.0, min(1.0, float(w))) for w in weights]
    if channel_count <= len(current):
        return current[:channel_count]
    return current + [1.0] * (channel_count - len(current))


def channel_display_name(channel: AcdcData, index: int) -> str:
    if channel.name:
        return channel.name
    return f"Channel {index + 1}"


def refresh_channel_display_levels(
    channels: Sequence[AcdcData],
) -> tuple[list[tuple[float, float] | None], list[tuple[float, float] | None]]:
    stack_levels: list[tuple[float, float] | None] = []
    display_clim: list[tuple[float, float] | None] = []
    for channel in channels:
        levels, clim = stack_display_levels(channel.image, channel.stack_shape)
        stack_levels.append(levels)
        display_clim.append(clim)
    return stack_levels, display_clim
