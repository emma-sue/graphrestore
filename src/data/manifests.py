"""Strict manifest parsing and degradation-name normalization.

The frozen training manifests deliberately live outside the MiO100 data root.
All relative training paths are therefore resolved against ``training_data_root``
and are required to stay inside the two approved MiOIR directories.  This
module never scans RAR, MiO100 exploration, or formal test image directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utils.hashing import is_sha256
from src.utils.io import iter_jsonl
from src.utils.paths import ensure_within

# This order is normative (V7.1 section 0.2) and is shared with the network.
SKILLS: tuple[str, ...] = (
    "noise",
    "motion_blur",
    "defocus_blur",
    "jpeg_artifact",
    "rain",
    "haze",
    "low_light",
    "low_resolution",
)
SKILL_TO_ID: dict[str, int] = {name: index for index, name in enumerate(SKILLS)}

MANIFEST_TO_SKILL: dict[str, str] = {
    "noise": "noise",
    "motion blur": "motion_blur",
    "defocus blur": "defocus_blur",
    "jpeg compression artifact": "jpeg_artifact",
    "rain": "rain",
    "haze": "haze",
    "dark": "low_light",
    "low resolution": "low_resolution",
}
SKILL_TO_MANIFEST: dict[str, str] = {
    value: key for key, value in MANIFEST_TO_SKILL.items()
}

ALLOWED_GROUP_A: tuple[tuple[str, str], ...] = (
    ("rain", "haze"),
    ("motion blur", "low resolution"),
    ("dark", "noise"),
    ("defocus blur", "jpeg compression artifact"),
    ("noise", "jpeg compression artifact"),
    ("rain", "low resolution"),
    ("motion blur", "dark"),
    ("defocus blur", "haze"),
)
ALLOWED_SINGLE: tuple[tuple[str], ...] = tuple(
    (name,) for name in MANIFEST_TO_SKILL
)
ALLOWED_PRIMARY_ORDERS = frozenset((*ALLOWED_SINGLE, *ALLOWED_GROUP_A))


class ManifestContractError(ValueError):
    """A manifest violates the frozen GraphRestore data contract."""


def normalize_skill_name(name: str) -> str:
    """Map a manifest/operator spelling to the normative skill name."""

    try:
        return MANIFEST_TO_SKILL[name]
    except KeyError as exc:
        raise ManifestContractError(f"unknown degradation name: {name!r}") from exc


def manifest_skill_name(name: str) -> str:
    """Map a normative skill name back to the official manifest spelling."""

    try:
        return SKILL_TO_MANIFEST[name]
    except KeyError as exc:
        raise ManifestContractError(f"unknown skill name: {name!r}") from exc


def _require_string(row: Mapping[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestContractError(f"{context}: {key} must be a non-empty string")
    return value


def _require_integer(row: Mapping[str, Any], key: str, *, context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestContractError(f"{context}: {key} must be an integer")
    return value


def _resolve_approved_file(
    relative_path: str,
    *,
    root: Path,
    approved_root: Path,
    context: str,
    must_exist: bool,
) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ManifestContractError(
            f"{context}: training paths must be relative to {root}: {candidate}"
        )
    resolved = ensure_within(root / candidate, approved_root)
    if must_exist and not resolved.is_file():
        raise ManifestContractError(f"{context}: referenced file is missing: {resolved}")
    return resolved


@dataclass(frozen=True)
class CleanRecord:
    clean_id: str
    clean_path: Path
    depth_path: Path
    clean_sha256: str
    width: int
    height: int
    depth_width: int
    depth_height: int
    split: str
    split_seed: int
    source: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class OperatorParameter:
    """One official operator invocation recorded by the frozen recipe."""

    name: str
    seed: int
    actual: Mapping[str, Any]

    @property
    def skill_name(self) -> str:
        return normalize_skill_name(self.name)

    @property
    def skill_id(self) -> int:
        return SKILL_TO_ID[self.skill_name]


@dataclass(frozen=True)
class PrimaryRecipe:
    sample_id: str
    split: str
    clean_id: str
    clean_path: Path
    depth_path: Path | None
    clean_sha256: str
    group: str
    seed: int
    operator_params: tuple[OperatorParameter, ...]
    raw: Mapping[str, Any]

    @property
    def operator_order(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.operator_params)

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(parameter.skill_name for parameter in self.operator_params)

    @property
    def skill_ids(self) -> tuple[int, ...]:
        return tuple(parameter.skill_id for parameter in self.operator_params)

    @property
    def is_pair(self) -> bool:
        return len(self.operator_params) == 2

    @property
    def contains_low_resolution(self) -> bool:
        return "low resolution" in self.operator_order


def load_clean_manifest(
    manifest_path: str | Path,
    training_data_root: str | Path,
    *,
    expected_split: str | None = None,
    must_exist: bool = True,
) -> tuple[CleanRecord, ...]:
    """Load a clean manifest and fail closed on path/split ambiguity."""

    root = Path(training_data_root).resolve()
    clean_root = (root / "source_clean" / "mioir_gt" / "GT").resolve()
    depth_root = (root / "depth" / "depth").resolve()
    records: list[CleanRecord] = []
    seen: set[str] = set()
    for line_number, row in iter_jsonl(manifest_path):
        context = f"{Path(manifest_path)}:{line_number}"
        clean_id = _require_string(row, "clean_id", context=context)
        if clean_id in seen:
            raise ManifestContractError(f"{context}: duplicate clean_id {clean_id!r}")
        seen.add(clean_id)
        split = _require_string(row, "split", context=context)
        if split not in {"train", "val"}:
            raise ManifestContractError(f"{context}: invalid clean split {split!r}")
        if expected_split is not None and split != expected_split:
            raise ManifestContractError(
                f"{context}: expected split {expected_split!r}, got {split!r}"
            )
        clean_path = _resolve_approved_file(
            _require_string(row, "clean_path", context=context),
            root=root,
            approved_root=clean_root,
            context=context,
            must_exist=must_exist,
        )
        depth_path = _resolve_approved_file(
            _require_string(row, "depth_path", context=context),
            root=root,
            approved_root=depth_root,
            context=context,
            must_exist=must_exist,
        )
        if clean_path.stem != clean_id or depth_path.stem != clean_id:
            raise ManifestContractError(
                f"{context}: clean/depth filename does not match clean_id {clean_id!r}"
            )
        clean_sha256 = _require_string(row, "clean_sha256", context=context)
        if not is_sha256(clean_sha256):
            raise ManifestContractError(f"{context}: malformed clean_sha256")
        width = _require_integer(row, "width", context=context)
        height = _require_integer(row, "height", context=context)
        depth_width = _require_integer(row, "depth_width", context=context)
        depth_height = _require_integer(row, "depth_height", context=context)
        if width <= 0 or height <= 0 or width % 4 or height % 4:
            raise ManifestContractError(
                f"{context}: clean dimensions must be positive multiples of four"
            )
        if (depth_width, depth_height) != (width // 4, height // 4):
            raise ManifestContractError(f"{context}: depth dimensions are not 1/4 clean")
        source = _require_string(row, "source", context=context)
        if source != "mioir_official_gt_depth":
            raise ManifestContractError(f"{context}: unapproved clean source {source!r}")
        records.append(
            CleanRecord(
                clean_id=clean_id,
                clean_path=clean_path,
                depth_path=depth_path,
                clean_sha256=clean_sha256,
                width=width,
                height=height,
                depth_width=depth_width,
                depth_height=depth_height,
                split=split,
                split_seed=_require_integer(row, "split_seed", context=context),
                source=source,
                raw=dict(row),
            )
        )
    if not records:
        raise ManifestContractError(f"empty clean manifest: {manifest_path}")
    return tuple(records)


def _parse_operator_parameters(
    row: Mapping[str, Any], *, context: str
) -> tuple[OperatorParameter, ...]:
    order = row.get("operator_order")
    degradations = row.get("degradations")
    parameters = row.get("operator_params")
    if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
        raise ManifestContractError(f"{context}: operator_order must be a string list")
    if degradations != order:
        raise ManifestContractError(f"{context}: degradations/operator_order mismatch")
    order_tuple = tuple(order)
    if order_tuple not in ALLOWED_PRIMARY_ORDERS:
        raise ManifestContractError(
            f"{context}: Group B/C or unknown operator order is forbidden: {order_tuple}"
        )
    if not isinstance(parameters, list) or len(parameters) != len(order):
        raise ManifestContractError(f"{context}: operator_params length mismatch")
    parsed: list[OperatorParameter] = []
    for ordinal, (name, parameter) in enumerate(zip(order, parameters)):
        if not isinstance(parameter, Mapping):
            raise ManifestContractError(
                f"{context}: operator_params[{ordinal}] must be an object"
            )
        if parameter.get("name") != name:
            raise ManifestContractError(
                f"{context}: operator_params[{ordinal}] name/order mismatch"
            )
        seed = parameter.get("seed")
        actual = parameter.get("actual")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ManifestContractError(
                f"{context}: operator_params[{ordinal}].seed is invalid"
            )
        if not isinstance(actual, Mapping):
            raise ManifestContractError(
                f"{context}: operator_params[{ordinal}].actual must be an object"
            )
        normalize_skill_name(name)
        parsed.append(OperatorParameter(name=name, seed=seed, actual=dict(actual)))
    return tuple(parsed)


def load_primary_manifest(
    manifest_path: str | Path,
    training_data_root: str | Path,
    *,
    expected_split: str | None = None,
    must_exist: bool = True,
) -> tuple[PrimaryRecipe, ...]:
    """Load only frozen single/Group-A MiOIR recipes.

    Any absolute training path, Group B/C order, MiO100 source, or RAR-style
    source fails before an image is opened.
    """

    root = Path(training_data_root).resolve()
    clean_root = (root / "source_clean" / "mioir_gt" / "GT").resolve()
    depth_root = (root / "depth" / "depth").resolve()
    records: list[PrimaryRecipe] = []
    seen: set[str] = set()
    for line_number, row in iter_jsonl(manifest_path):
        context = f"{Path(manifest_path)}:{line_number}"
        sample_id = _require_string(row, "sample_id", context=context)
        if sample_id in seen:
            raise ManifestContractError(f"{context}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        split = _require_string(row, "split", context=context)
        if split not in {"train", "val"}:
            raise ManifestContractError(f"{context}: invalid primary split {split!r}")
        if expected_split is not None and split != expected_split:
            raise ManifestContractError(
                f"{context}: expected split {expected_split!r}, got {split!r}"
            )
        source = _require_string(row, "source", context=context)
        if source != "agenticir_official":
            raise ManifestContractError(f"{context}: unapproved recipe source {source!r}")
        clean_id = _require_string(row, "clean_id", context=context)
        clean_path = _resolve_approved_file(
            _require_string(row, "clean_path", context=context),
            root=root,
            approved_root=clean_root,
            context=context,
            must_exist=must_exist,
        )
        if clean_path.stem != clean_id:
            raise ManifestContractError(f"{context}: clean_path/clean_id mismatch")
        operator_params = _parse_operator_parameters(row, context=context)
        order = tuple(parameter.name for parameter in operator_params)
        group = _require_string(row, "group", context=context)
        expected_group = "single" if len(order) == 1 else "A"
        if group != expected_group:
            raise ManifestContractError(
                f"{context}: expected group {expected_group!r}, got {group!r}"
            )
        raw_depth = row.get("depth_path")
        if "haze" in order:
            if not isinstance(raw_depth, str) or not raw_depth:
                raise ManifestContractError(f"{context}: haze recipe lacks depth_path")
            depth_path: Path | None = _resolve_approved_file(
                raw_depth,
                root=root,
                approved_root=depth_root,
                context=context,
                must_exist=must_exist,
            )
            if depth_path.stem != clean_id:
                raise ManifestContractError(f"{context}: depth_path/clean_id mismatch")
        else:
            if raw_depth is not None:
                raise ManifestContractError(f"{context}: non-haze recipe has depth_path")
            depth_path = None
        clean_sha256 = _require_string(row, "clean_sha256", context=context)
        if not is_sha256(clean_sha256):
            raise ManifestContractError(f"{context}: malformed clean_sha256")
        recipe_seed = _require_integer(row, "seed", context=context)
        if not 0 <= recipe_seed < 2**32:
            raise ManifestContractError(f"{context}: recipe seed is outside uint32")
        records.append(
            PrimaryRecipe(
                sample_id=sample_id,
                split=split,
                clean_id=clean_id,
                clean_path=clean_path,
                depth_path=depth_path,
                clean_sha256=clean_sha256,
                group=group,
                seed=recipe_seed,
                operator_params=operator_params,
                raw=dict(row),
            )
        )
    if not records:
        raise ManifestContractError(f"empty primary manifest: {manifest_path}")
    return tuple(records)


def task_buckets(records: Sequence[PrimaryRecipe]) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Return stable dataset-index buckets keyed by official operator order."""

    result: dict[tuple[str, ...], list[int]] = {}
    for index, record in enumerate(records):
        result.setdefault(record.operator_order, []).append(index)
    return {key: tuple(indices) for key, indices in result.items()}


def assert_disjoint_clean_ids(
    left: Iterable[CleanRecord | PrimaryRecipe],
    right: Iterable[CleanRecord | PrimaryRecipe],
) -> None:
    """Raise when two record collections share a clean ID."""

    overlap = {record.clean_id for record in left} & {record.clean_id for record in right}
    if overlap:
        preview = ", ".join(sorted(overlap)[:8])
        raise ManifestContractError(f"clean-ID overlap detected: {preview}")

