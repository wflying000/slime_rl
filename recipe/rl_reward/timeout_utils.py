"""Timeout propagation and the invalid-reward contract shared by all scorers."""

from __future__ import annotations

from typing import Any


class RewardTimeout(BaseException):
    """Bypass legacy ``except Exception: return 0`` reward fallbacks."""


def is_timeout_error(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (RewardTimeout, TimeoutError)) or any(
            "timeout" in cls.__name__.lower() for cls in type(exc).__mro__
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def raise_if_timeout(exc: BaseException) -> None:
    if is_timeout_error(exc):
        raise RewardTimeout(f"{type(exc).__name__}: {exc}") from exc


def is_invalid_reward(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None)
    return bool(getattr(sample, "remove_sample", False)) or (
        isinstance(metadata, dict)
        and bool(metadata.get("reward_timeout") or metadata.get("reward_error"))
    )


def mark_invalid_reward(sample: Any, *, timeout: bool, reason: str) -> None:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata["reward_timeout" if timeout else "reward_error"] = True
    metadata["reward_invalid_reason"] = reason[:1000]
    sample.remove_sample = True
    # Set immediately as well as at the central training-data boundary.
    sample.loss_mask = [0] * getattr(sample, "response_length", 0)
