#!/usr/bin/env python3
"""Sidebar widget: on-beamline vs off-beamline data root.

The same proposal folder has two absolute paths — ``/nsls2/data1/<bl>/proposals``
at the beamline and ``/mnt/data32/NSLSII_Data/nsls2_romote/<bl>_remote`` through
the sshfs mount (see :mod:`NanoOrganizer.core.beamline_paths`). Pages call
:func:`site_toggle` once in the sidebar; it renders the radio + beamline picker
and returns the resolved site key, and :func:`apply_site` rewrites any default
or user-typed path onto that site.

Drop-in use in a page::

    site, bl = site_toggle("tsaxs")
    default = apply_site(DEFAULT_ANALYSIS, site, bl)
    analysis = folder_picker(key="tsaxs_analysis", default=default, ...)

``apply_site`` leaves unrecognised paths (anything not under a known beamline
root) untouched, so local/test folders keep working unchanged.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from NanoOrganizer.core.beamline_paths import (
    BEAMLINES, SITE_KEYS, dataset_path, detect_site, resolve_site, site_label,
    site_root, split_dataset_path, swap_site,
)
from NanoOrganizer.core.access_config import configured_beamline, configured_site

__all__ = ["site_toggle", "apply_site", "follow_site", "dataset_path_picker"]

# Radio options: "Auto" first so the common case needs no interaction.
_AUTO = "Auto (detect)"
_OPTIONS = [_AUTO] + [site_label(s) for s in SITE_KEYS]
_LABEL_TO_KEY = {site_label(s): s for s in SITE_KEYS}


def site_toggle(key: str, beamline: str = "smi", show_beamline: bool = True,
                expanded: bool = False):
    """Render the site/beamline chooser. Returns ``(site_key, beamline)``.

    ``site_key`` is always concrete (``"onsite"`` / ``"offsite"``) — ``Auto``
    is resolved here so callers never have to.

    Parameters
    ----------
    key : str        Unique prefix for this widget's session-state keys.
    beamline : str   Default beamline ('smi' / 'cms').
    show_beamline : bool  Hide the beamline picker for single-beamline pages.
    expanded : bool  Start the expander open.
    """
    with st.expander("🛰️ Data location", expanded=expanded):
        configured = configured_site()
        configured_choice = _AUTO
        if configured in _LABEL_TO_KEY.values():
            configured_choice = next(
                label for label, key in _LABEL_TO_KEY.items() if key == configured
            )
        choice_index = _OPTIONS.index(configured_choice)
        choice = st.radio(
            "Where is this running?", _OPTIONS, index=choice_index,
            key=f"{key}_site_choice", horizontal=True,
            help="Auto picks whichever data tree is actually mounted here. "
                 "Pin it with $NANOORGANIZER_SITE or pyViz.conf if both are visible.")

        bl = beamline
        if show_beamline:
            options = list(BEAMLINES)
            configured_bl = configured_beamline()
            selected_bl = beamline
            if beamline == "smi" and configured_bl in options:
                selected_bl = configured_bl
            idx = options.index(selected_bl) if selected_bl in options else 0
            bl = st.selectbox("Beamline", options, index=idx,
                              key=f"{key}_beamline",
                              format_func=str.upper)

        if choice == _AUTO:
            detected = detect_site(bl)
            site = resolve_site("auto", bl)
            if detected is None:
                st.warning(
                    "Neither data tree is mounted here — falling back to "
                    f"**{site_label(site)}**. Paths below may not resolve.")
            else:
                st.caption(f"Detected: **{site_label(site)}**")
        else:
            site = _LABEL_TO_KEY[choice]

        root = site_root(site, bl)
        exists = Path(root).is_dir()
        st.code(str(root), language="bash")
        st.caption(("✅ mounted" if exists else "⚠️ not present on this machine")
                   + f" · beamline {bl.upper()}")

    return site, bl


def apply_site(path, site: str, beamline: str = "smi"):
    """Re-root ``path`` onto ``site``, or return it unchanged if unrecognised.

    Used to make a page's hard-coded ``DEFAULT_*`` path follow the toggle: the
    default is written once (in either spelling) and this converts it.
    """
    if not path:
        return path
    swapped = swap_site(path, site)
    return str(swapped) if swapped is not None else str(path)


def follow_site(picker_key: str, site: str, beamline: str = "smi") -> None:
    """Keep a :func:`folder_picker`'s remembered path on the selected site.

    ``apply_site`` only fixes the *default*; once the user has browsed, the
    chosen folder lives in ``st.session_state[f"{picker_key}_path"]`` and would
    otherwise stay on the old root after flipping the toggle. Call this right
    after :func:`site_toggle` and before the picker is rendered.
    """
    state_key = f"{picker_key}_path"
    last_key = f"{picker_key}_site_last"
    previous = st.session_state.get(last_key)
    st.session_state[last_key] = site
    if previous is None or previous == site:
        return
    current = st.session_state.get(state_key)
    if not current:
        return
    swapped = swap_site(current, site)
    if swapped is not None:
        st.session_state[state_key] = str(swapped)


def dataset_path_picker(key: str, cycle: str = "2026-2",
                        proposal: str = "pass-317378",
                        project: str = "", results: str = "",
                        beamline: str = "smi", site: str = "auto"):
    """Compose a dataset path from its parts, honouring the site toggle.

    A companion to :func:`site_toggle` for pages that would rather have the
    user type ``cycle`` / ``proposal`` / ``project`` than a full path. Returns
    the assembled path as a string.
    """
    c1, c2 = st.columns(2)
    cyc = c1.text_input("Cycle", value=cycle, key=f"{key}_cycle")
    prop = c2.text_input("Proposal", value=proposal, key=f"{key}_proposal")
    c3, c4 = st.columns(2)
    proj = c3.text_input("Project", value=project, key=f"{key}_project")
    res = c4.text_input("Results subfolder", value=results, key=f"{key}_results",
                        help="e.g. tsaxs, twaxs, maxs — leave blank for the "
                             "project's Results/ folder.")
    return str(dataset_path(cyc, prop, project=proj, results=res,
                            beamline=beamline, site=site))
