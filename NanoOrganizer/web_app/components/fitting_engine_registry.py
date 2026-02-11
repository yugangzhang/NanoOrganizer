#!/usr/bin/env python3
"""
1D fitting engine registry for NanoOrganizer.

This module provides a lightweight plugin-style interface to register fitting
engines and their configuration-schema stubs without modifying page code.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

EngineStatus = Literal["ready", "planned", "experimental", "disabled"]
ParameterType = Literal["float", "int", "bool", "select", "text"]


@dataclass(frozen=True)
class ParameterSpec:
    """Schema descriptor for one engine configuration parameter."""

    name: str
    param_type: ParameterType
    label: str
    default: Any = None
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    options: tuple[str, ...] = field(default_factory=tuple)
    help_text: str = ""
    section: str = "General"


@dataclass(frozen=True)
class EngineSpec:
    """Metadata record for one fitting engine."""

    key: str
    label: str
    status: EngineStatus
    domain: str
    description: str
    backend: Optional[str] = None
    supports_batch: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_ENGINE_SPECS: list[EngineSpec] = [
    EngineSpec(
        key="general_peaks",
        label="General Peaks",
        status="ready",
        domain="Generic 1D",
        description="Multi-peak fitting with Gaussian, Lorentzian, and Pseudo-Voigt shapes.",
        backend="general_peaks",
        supports_batch=True,
        tags=("multi_peak", "deconvolution"),
    ),
    EngineSpec(
        key="saxs_physics",
        label="SAXS Physics",
        status="ready",
        domain="SAXS S(q)",
        description="Form-factor fitting with shape, polydispersity, and optional Porod term.",
        backend="saxs_physics",
        supports_batch=True,
        tags=("physics", "form_factor"),
    ),
    EngineSpec(
        key="xas",
        label="XAS (planned)",
        status="planned",
        domain="XAS",
        description="Planned: edge-step, white-line, and EXAFS-oriented model routes.",
        supports_batch=True,
        tags=("xas",),
    ),
    EngineSpec(
        key="xps",
        label="XPS (planned)",
        status="planned",
        domain="XPS",
        description="Planned: constrained multi-component core-level peak modeling.",
        supports_batch=True,
        tags=("xps",),
    ),
    EngineSpec(
        key="uv_vis",
        label="UV-Vis (planned)",
        status="planned",
        domain="UV-Vis",
        description="Planned: baseline + band deconvolution and kinetic trend fitting.",
        supports_batch=True,
        tags=("uv_vis",),
    ),
    EngineSpec(
        key="xpcs",
        label="XPCS (planned)",
        status="planned",
        domain="XPCS",
        description="Planned: g2/tau model fitting with dynamics constraints.",
        supports_batch=True,
        tags=("xpcs",),
    ),
]

DEFAULT_ENGINE_SCHEMAS: dict[str, list[ParameterSpec]] = {
    "general_peaks": [
        ParameterSpec(
            name="shape",
            param_type="select",
            label="Peak shape",
            default="Pseudo-Voigt",
            options=("Gaussian", "Lorentzian", "Pseudo-Voigt"),
            section="Model",
        ),
        ParameterSpec(
            name="default_peak_count",
            param_type="int",
            label="Default peaks",
            default=2,
            min_value=1,
            max_value=12,
            step=1,
            section="Model",
        ),
        ParameterSpec(
            name="max_iterations",
            param_type="int",
            label="Max iterations",
            default=2000,
            min_value=100,
            max_value=10000,
            step=100,
            section="Solver",
        ),
    ],
    "saxs_physics": [
        ParameterSpec(
            name="shape",
            param_type="select",
            label="Form-factor shape",
            default="sphere",
            options=("sphere", "cube", "octahedron"),
            section="Model",
        ),
        ParameterSpec(
            name="polydisperse",
            param_type="bool",
            label="Polydisperse",
            default=False,
            section="Model",
        ),
        ParameterSpec(
            name="use_porod",
            param_type="bool",
            label="Use Porod term",
            default=False,
            section="Model",
        ),
        ParameterSpec(
            name="max_iterations",
            param_type="int",
            label="Max iterations",
            default=2000,
            min_value=100,
            max_value=20000,
            step=100,
            section="Solver",
        ),
    ],
    "xas": [
        ParameterSpec(
            name="edge_energy",
            param_type="float",
            label="Edge energy (stub)",
            default=0.0,
            section="Model",
            help_text="Placeholder schema for future XAS plugin integration.",
        )
    ],
    "xps": [
        ParameterSpec(
            name="background_model",
            param_type="select",
            label="Background model (stub)",
            default="Shirley",
            options=("Shirley", "Tougaard"),
            section="Model",
            help_text="Placeholder schema for future XPS plugin integration.",
        )
    ],
    "uv_vis": [
        ParameterSpec(
            name="baseline_model",
            param_type="select",
            label="Baseline model (stub)",
            default="Polynomial",
            options=("Constant", "Linear", "Polynomial"),
            section="Model",
            help_text="Placeholder schema for future UV-Vis plugin integration.",
        )
    ],
    "xpcs": [
        ParameterSpec(
            name="g2_model",
            param_type="select",
            label="g2 model (stub)",
            default="SingleExp",
            options=("SingleExp", "StretchedExp", "DoubleExp"),
            section="Model",
            help_text="Placeholder schema for future XPCS plugin integration.",
        )
    ],
}

_ENGINE_SPECS_BY_KEY: Dict[str, EngineSpec] = {spec.key: spec for spec in DEFAULT_ENGINE_SPECS}
_ENGINE_SCHEMAS_BY_KEY: Dict[str, list[ParameterSpec]] = {
    key: list(schema) for key, schema in DEFAULT_ENGINE_SCHEMAS.items()
}


def register_engine(
    spec: EngineSpec,
    *,
    config_schema: Optional[Sequence[ParameterSpec]] = None,
    overwrite: bool = False,
) -> None:
    """Register a new engine (or overwrite an existing one if allowed)."""
    if spec.key in _ENGINE_SPECS_BY_KEY and not overwrite:
        raise ValueError(f"Engine `{spec.key}` is already registered.")
    _ENGINE_SPECS_BY_KEY[spec.key] = spec
    if config_schema is not None:
        _ENGINE_SCHEMAS_BY_KEY[spec.key] = list(config_schema)
    elif spec.key not in _ENGINE_SCHEMAS_BY_KEY:
        _ENGINE_SCHEMAS_BY_KEY[spec.key] = []


def list_engine_specs(*, status: Optional[EngineStatus] = None, include_disabled: bool = False) -> list[EngineSpec]:
    """Return registered engine specs with optional status filtering."""
    specs = list(_ENGINE_SPECS_BY_KEY.values())
    if status is not None:
        return [spec for spec in specs if spec.status == status]
    if include_disabled:
        return specs
    return [spec for spec in specs if spec.status != "disabled"]


def list_engine_rows(*, status: Optional[EngineStatus] = None, include_disabled: bool = False) -> list[dict[str, Any]]:
    """Return engine metadata as dict rows for table display."""
    rows = []
    for spec in list_engine_specs(status=status, include_disabled=include_disabled):
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "backend": spec.backend,
                "status": spec.status,
                "domain": spec.domain,
                "supports_batch": spec.supports_batch,
                "description": spec.description,
                "tags": ", ".join(spec.tags),
            }
        )
    return rows


def get_engine_spec(engine_key: str) -> Optional[EngineSpec]:
    """Get one engine spec by key."""
    return _ENGINE_SPECS_BY_KEY.get(str(engine_key))


def get_engine_schema(engine_key: str) -> list[ParameterSpec]:
    """Get config-schema stubs for an engine key."""
    return list(_ENGINE_SCHEMAS_BY_KEY.get(str(engine_key), []))


def get_engine_schema_rows(engine_key: str) -> list[dict[str, Any]]:
    """Return config-schema rows for quick UI previews/documentation."""
    rows = []
    for param in get_engine_schema(engine_key):
        rows.append(
            {
                "section": param.section,
                "name": param.name,
                "label": param.label,
                "type": param.param_type,
                "default": param.default,
                "min": param.min_value,
                "max": param.max_value,
                "options": ", ".join(param.options),
                "help": param.help_text,
            }
        )
    return rows


def get_ready_backend_labels(backend_label_to_key: dict[str, str]) -> list[str]:
    """
    Return ready engine labels constrained to currently implemented backends.

    `backend_label_to_key` keeps UI labels stable in the page while the registry
    controls what is currently ready/planned.
    """
    ready_labels = {spec.label for spec in list_engine_specs(status="ready")}
    return [label for label in backend_label_to_key if label in ready_labels]
