"""Middleware composition (Gin-style context + next)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from acdc.middleware.context import AcdcContext

AcdcMiddleware: TypeAlias = Callable[[AcdcContext, Callable[[], None]], None]


def use(*middleware: AcdcMiddleware) -> Callable[[AcdcContext], AcdcContext]:
    """Compose middleware and return a runner."""

    def run(ctx: AcdcContext) -> AcdcContext:
        stack = list(middleware)

        def dispatch(index: int) -> None:
            if index >= len(stack):
                return

            def next_() -> None:
                dispatch(index + 1)

            stack[index](ctx, next_)

        dispatch(0)
        return ctx

    return run
