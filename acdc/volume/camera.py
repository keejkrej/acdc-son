"""3D volume viewer camera — hand-tool navigation aligned with the 2D viewer."""

from __future__ import annotations

import numpy as np
from vispy.scene.cameras import TurntableCamera
from vispy.util import keys

_LEFT_BUTTON = 1
_RIGHT_BUTTON = 2
_MIDDLE_BUTTON = 3

# Look down the Z axis (XY plane on screen, like the 2D segment viewer).
DEFAULT_ELEVATION = -90.0
DEFAULT_AZIMUTH = 0.0
DEFAULT_ROLL = 0.0
DEFAULT_FOV = 0.0  # orthographic — no perspective tilt


class VolumeTurntableCamera(TurntableCamera):
    """Hand tool: drag to pan, scroll to zoom; Shift+drag to orbit (3D only)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("fov", DEFAULT_FOV)
        kwargs.setdefault("elevation", DEFAULT_ELEVATION)
        kwargs.setdefault("azimuth", DEFAULT_AZIMUTH)
        kwargs.setdefault("roll", DEFAULT_ROLL)
        super().__init__(**kwargs)
        self._space_pan = False

    @property
    def space_pan(self) -> bool:
        return self._space_pan

    @space_pan.setter
    def space_pan(self, value: bool) -> None:
        self._space_pan = bool(value)

    def _should_pan(self, buttons: list[int], modifiers: tuple[str, ...]) -> bool:
        if keys.SHIFT in modifiers:
            return False
        if _MIDDLE_BUTTON in buttons:
            return True
        if self._space_pan and _LEFT_BUTTON in buttons:
            return True
        if _LEFT_BUTTON in buttons and _RIGHT_BUTTON in buttons:
            return False
        return _LEFT_BUTTON in buttons or _RIGHT_BUTTON in buttons

    def _should_orbit(self, buttons: list[int], modifiers: tuple[str, ...]) -> bool:
        return (
            _LEFT_BUTTON in buttons
            and keys.SHIFT in modifiers
            and _RIGHT_BUTTON not in buttons
        )

    def _pan_center(self, p1, p2) -> None:
        if self._viewbox is None:
            return
        norm = float(np.mean(self._viewbox.size))
        if norm <= 0:
            return
        if self._event_value is None or len(self._event_value) == 2:
            self._event_value = self.center
        dist = (p1 - p2) / norm * self._scale_factor
        dist[1] *= -1
        dx, dy, dz = self._dist_to_trans(dist)
        flip = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = flip[0] * dx, flip[1] * dy, dz * flip[2]
        center = self._event_value
        self.center = center[0] + dx, center[1] + dy, center[2] + dz

    def viewbox_mouse_event(self, event) -> None:
        if event.handled or not self.interactive:
            return

        if event.type == "mouse_release":
            self._event_value = None

        if event.type in ("mouse_press", "mouse_move"):
            buttons = event.buttons or []
            modifiers = event.mouse_event.modifiers

            if self._should_orbit(buttons, modifiers):
                if event.type == "mouse_press":
                    event.handled = True
                    self._event_value = None
                    return
                if event.type == "mouse_move" and event.press_event is not None:
                    self._update_rotation(event)
                    event.handled = True
                    return

            if self._should_pan(buttons, modifiers):
                if event.type == "mouse_press":
                    event.handled = True
                    self._event_value = None
                    return
                if event.type == "mouse_move" and event.press_event is not None:
                    press = event.mouse_event.press_event.pos
                    pos = event.mouse_event.pos
                    self._pan_center(press, pos)
                    event.handled = True
                    return

        super().viewbox_mouse_event(event)
