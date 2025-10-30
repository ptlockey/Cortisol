"""Custom Streamlit component for editing floorplan schematics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_COMPONENT_NAME = "floorplan_editor"
_COMPONENT_DIR = Path(__file__).parent
_FRONTEND_DIR = _COMPONENT_DIR.parent / "frontend"
_BUILD_DIR = _FRONTEND_DIR / "build"


if _BUILD_DIR.exists():
    _component_func = components.declare_component(
        _COMPONENT_NAME, path=str(_BUILD_DIR)
    )
else:
    # Fall back to the Streamlit dev server behaviour by expecting a local dev server
    # at the URL defined via the STREAMLIT_COMPONENT_URL environment variable.
    dev_url = os.environ.get("STREAMLIT_COMPONENT_URL", "http://localhost:3001")
    _component_func = components.declare_component(_COMPONENT_NAME, url=dev_url)


def floorplan_editor(
    *,
    data: Dict[str, Any],
    key: Optional[str] = None,
    disabled: bool = False,
    height: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Render the floorplan editor component.

    Parameters
    ----------
    data:
        The initial JSON-serialisable data to display in the editor.
    key:
        Optional Streamlit widget key to keep the component state.
    disabled:
        When true, the editor enters a read-only mode.
    height:
        Optional height hint for the underlying iframe. The component
        dynamically resizes itself as the user interacts with it, but providing
        a hint helps avoid layout jumps when first rendered.
    kwargs:
        Additional keyword arguments forwarded to the component (for example,
        configuration flags for snapping).

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary describing the delta emitted by the component, or ``None``
        if nothing has changed.
    """

    component_value = _component_func(
        data=data,
        disabled=disabled,
        height=height,
        default=None,
        key=key,
        **kwargs,
    )
    if component_value is None:
        return None
    if isinstance(component_value, str):
        try:
            return json.loads(component_value)
        except json.JSONDecodeError:
            return None
    return component_value


__all__ = ["floorplan_editor"]
