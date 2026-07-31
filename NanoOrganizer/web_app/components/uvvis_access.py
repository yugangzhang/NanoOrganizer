#!/usr/bin/env python3
"""Owner-only access gate for the UV-Vis explorer pages.

The two UV-Vis pages (feature maps + 1-D spectra) are private to a single owner.
This helper only enforces that in **multi-user** secure mode
(``NANOORGANIZER_USERS_FILE`` set): only the configured owner (or an admin) may
view them; every other logged-in user gets a stop.  In the other two deployments
the pages are open, because there is no per-user identity to check:

* **single-password** secure mode (``./run PORT PASSWORD`` with no users file) —
  one shared password gates the whole app, so whoever unlocked it is trusted.
* **no security** (local single-user use) — the owner is on their own machine.

Owner is read from ``NANOORGANIZER_UVVIS_OWNER`` (default ``"yuzhang"``), matched
case-insensitively against the logged-in username.
"""

from __future__ import annotations

import os

import streamlit as st

DEFAULT_OWNER = "yuzhang"


def require_uvvis_owner() -> None:
    """Stop the page unless the current user owns the UV-Vis pages.

    Only enforced in **multi-user** secure mode (a users file is configured):
    then only the owner username (case-insensitive) or an admin passes.  In
    single-password secure mode and with no security, there is no per-user
    identity to check, so the pages are open (the shared password / local
    machine is the access control).
    """
    owner = os.environ.get("NANOORGANIZER_UVVIS_OWNER", DEFAULT_OWNER).strip().lower()

    try:
        from NanoOrganizer.web_app.components.security import (
            current_user, is_admin,
        )
    except Exception:
        # security helpers unavailable (standalone) → treat as open local use
        return

    user = current_user()
    if not user:
        # no per-user login (single-password mode or unrestricted) → open
        return

    if user.strip().lower() == owner or is_admin():
        return

    st.error("🔒 The UV-Vis explorer pages are private to their owner.")
    st.caption("Ask the owner to grant access, or sign in as the owner account.")
    st.stop()
