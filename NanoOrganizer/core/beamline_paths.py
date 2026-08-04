#!/usr/bin/env python3
"""Site-aware NSLS-II data roots — one dataset, two filesystems.

The same proposal folder is reached by two different absolute paths depending
on where the code runs:

* **on-beamline** (a beamline workstation / the NSLS-II cluster)::

      /nsls2/data1/<beamline>/proposals/<cycle>/<proposal>/projects/...

* **off-beamline** (the lab machine, through the sshfs mount)::

      /mnt/data32/NSLSII_Data/nsls2_romote/<beamline>_remote/<cycle>/<proposal>/projects/...

Everything below ``<cycle>/<proposal>/`` is identical, so this module splits a
dataset location into a *site root* (which changes) and a *relative suffix*
(which does not). Point notebooks and GUI pages at
:func:`dataset_path` / :func:`resolve_site` and switching machines becomes a
one-line toggle instead of an edit-every-default hunt.

Typical use::

    from NanoOrganizer.core.beamline_paths import dataset_path

    # auto: picks whichever site actually exists on this machine
    p = dataset_path("2026-2", "pass-317378",
                     project="Digestive_Ripening", results="tsaxs")

    # forced
    p = dataset_path("2026-2", "pass-317378", beamline="smi", site="onsite", ...)

The site keys are ``"onsite"`` (at the beamline) and ``"offsite"`` (mounted
copy); ``"auto"`` picks the first one whose root directory exists.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "SITES", "SITE_KEYS", "BEAMLINES",
    "site_label", "site_root", "detect_site", "resolve_site",
    "proposal_path", "dataset_path", "candidate_roots", "swap_site",
    "split_dataset_path",
]

# Environment overrides — set NANOORGANIZER_SITE=onsite|offsite to pin the site
# on a machine where auto-detection would guess wrong (e.g. both trees mounted).
ENV_SITE = "NANOORGANIZER_SITE"
ENV_ONSITE_ROOT = "NANOORGANIZER_ONSITE_ROOT"
ENV_OFFSITE_ROOT = "NANOORGANIZER_OFFSITE_ROOT"

# Beamlines this maps for. The key is the lowercase beamline short name used in
# both path styles (/nsls2/data1/<key>/ and .../<key>_remote/).
BEAMLINES = ("smi", "cms")

# Per-site root *templates*. ``{bl}`` is substituted with the beamline key.
#
# ``roots`` is ordered: the first entry is canonical, the rest are equivalent
# spellings (symlinks) kept so an existing hard-coded default still resolves.
SITES = {
    "onsite": {
        "label": "On the beamline (/nsls2)",
        "hint": "Beamline workstation or the NSLS-II cluster.",
        "roots": [
            "/nsls2/data1/{bl}/proposals",
            "/nsls2/data/{bl}/proposals",
            "/nsls2/auto-storage/{bl}/proposals",
            # Personal symlinks used in the beamline configs/notebooks; both
            # spellings ("proposal" and "proposals") are in the wild.
            "/nsls2/users/yuzhang/{bl}_proposals_link",
            "/nsls2/users/yuzhang/{bl}_proposal_link",
        ],
    },
    "offsite": {
        "label": "Off-beamline (sshfs mount)",
        "hint": "Lab machine reading the mounted copy of the beamline data.",
        "roots": [
            "/mnt/data32/NSLSII_Data/nsls2_romote/{bl}_remote",
            "~/NSLSII_Data_Link/nsls2_romote/{bl}_remote",
        ],
    },
}

# Detection order: prefer the real beamline filesystem when both are visible.
SITE_KEYS = ("onsite", "offsite")


def site_label(site: str) -> str:
    """Human-readable name for a site key (falls back to the key itself)."""
    return SITES.get(site, {}).get("label", str(site))


def candidate_roots(site: str, beamline: str = "smi"):
    """All root spellings for ``site``/``beamline``, canonical first.

    Returns ``Path`` objects with ``~`` expanded. Existence is *not* checked —
    use :func:`site_root` for that.
    """
    if site not in SITES:
        raise ValueError(
            f"Unknown site {site!r}; expected one of {list(SITES)}.")
    bl = str(beamline).strip().lower()

    override = os.environ.get(
        ENV_ONSITE_ROOT if site == "onsite" else ENV_OFFSITE_ROOT, "").strip()
    templates = ([override] if override else []) + list(SITES[site]["roots"])
    return [Path(t.format(bl=bl)).expanduser() for t in templates]


def site_root(site: str, beamline: str = "smi", must_exist: bool = False):
    """The root folder for ``site``/``beamline``.

    Returns the first spelling that exists; if none does, returns the canonical
    one (so error messages show the path the user expected) unless
    ``must_exist`` is set, in which case ``None`` is returned.
    """
    roots = candidate_roots(site, beamline)
    for root in roots:
        if root.is_dir():
            return root
    return None if must_exist else roots[0]


def detect_site(beamline: str = "smi"):
    """Guess the site from what is actually mounted on this machine.

    Honours ``$NANOORGANIZER_SITE`` first. Returns a site key, or ``None`` when
    neither tree is visible (a caller can then fall back to plain browsing).
    """
    pinned = os.environ.get(ENV_SITE, "").strip().lower()
    if pinned in SITES:
        return pinned
    for site in SITE_KEYS:
        if site_root(site, beamline, must_exist=True) is not None:
            return site
    return None


def resolve_site(site: str = "auto", beamline: str = "smi") -> str:
    """Normalise a site argument, resolving ``"auto"`` via :func:`detect_site`.

    Falls back to ``"offsite"`` when nothing is mounted, since that is the
    spelling a lab machine would eventually use.
    """
    site = (site or "auto").strip().lower()
    if site == "auto":
        return detect_site(beamline) or "offsite"
    if site not in SITES:
        raise ValueError(
            f"Unknown site {site!r}; expected 'auto' or one of {list(SITES)}.")
    return site


def proposal_path(cycle: str, proposal: str, beamline: str = "smi",
                  site: str = "auto"):
    """``<root>/<cycle>/<proposal>`` for the resolved site.

    ``cycle`` is the NSLS-II cycle as it appears in the path (``"2026-2"``) and
    ``proposal`` the pass folder (``"pass-317378"``).
    """
    root = site_root(resolve_site(site, beamline), beamline)
    return root / str(cycle) / str(proposal)


def dataset_path(cycle: str, proposal: str, project: str = "",
                 results: str = "", beamline: str = "smi",
                 site: str = "auto", subdir: str = "Results"):
    """Full path to a project's reduced-data folder.

    ``<root>/<cycle>/<proposal>/projects/<project>/<subdir>/<results>``

    ``project`` and ``results`` are optional, so this also yields the
    ``projects/`` folder (both empty) or a whole project (``results=""``).
    Pass ``subdir=""`` to skip the ``Results`` level (e.g. for ``user_data``).
    """
    path = proposal_path(cycle, proposal, beamline, site)
    if project:
        path = path / "projects" / project
        if subdir:
            path = path / subdir
        if results:
            path = path / results
    elif not project and results:
        path = path / results
    return path


def split_dataset_path(path):
    """Inverse of :func:`dataset_path`: ``(site, beamline, suffix)``.

    ``suffix`` is the part below the site root (``"2026-2/pass-317378/..."``),
    or ``None`` for all three fields when ``path`` is not under a known root.
    Symlinks are resolved so ``~/NSLSII_Data_Link/...`` matches the real mount.
    """
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except OSError:                                # pragma: no cover - odd FS
        resolved = target
    for site in SITES:
        for bl in BEAMLINES:
            for root in candidate_roots(site, bl):
                for base in {root, _safe_resolve(root)}:
                    try:
                        rel = resolved.relative_to(base)
                    except ValueError:
                        try:
                            rel = target.relative_to(base)
                        except ValueError:
                            continue
                    return site, bl, str(rel)
    return None, None, None


def swap_site(path, site: str):
    """Re-root ``path`` onto another site, keeping the suffix.

    Returns ``None`` when ``path`` is not recognised as living under any known
    root — the caller should then leave the path untouched.
    """
    found_site, beamline, rel = split_dataset_path(path)
    if rel is None:
        return None
    target = resolve_site(site, beamline or "smi")
    if target == found_site:
        return Path(path).expanduser()
    return site_root(target, beamline or "smi") / rel


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:                                # pragma: no cover - odd FS
        return path
