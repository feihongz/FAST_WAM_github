"""Stable, physically verifiable identity for a LIBERO simulator runtime.

The Phase-A closed-loop evaluation depends on code and native libraries that
live outside the FAST-WAM Git tree.  This module records those dependencies in
one self-hashed JSON object and can recapture them later to fail closed on any
drift.  The LIBERO tree digest intentionally covers executable source and
physics/configuration files, not large meshes or textures; task BDDL and init
states are bound separately by the Phase-A task-source ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any


LIBERO_SIMULATOR_IDENTITY_SCHEMA_VERSION = 1
LIBERO_SIMULATOR_IDENTITY_KIND = "libero_simulator_runtime_identity"

_SOURCE_SUFFIXES = (
    ".bddl",
    ".cfg",
    ".ini",
    ".json",
    ".py",
    ".xml",
    ".yaml",
    ".yml",
)
_EXCLUDED_TREE_PARTS = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
_RUNTIME_ENVIRONMENT_KEYS = (
    "FASTWAM_LIBERO_ROOT",
    "LD_LIBRARY_PATH",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "PYTHONPATH",
)
_IMAGE_DIGEST_ENV_KEYS = (
    "FASTWAM_CONTAINER_IMAGE_DIGEST",
    "FASTWAM_JIHE_IMAGE_DIGEST",
    "JIHE_IMAGE_DIGEST",
    "CONTAINER_IMAGE_DIGEST",
)
_JIHE_ENVIRONMENT_ID_KEYS = (
    "FASTWAM_JIHE_ENVIRONMENT_IDENTITY",
    "JIHE_ENVIRONMENT_IDENTITY",
)
_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RENDERER_POLICY = (
    "egl_with_child_MUJOCO_EGL_DEVICE_ID_equal_to_assigned_"
    "physical_CUDA_VISIBLE_DEVICES_token"
)
_PACKAGE_NAMES = ("torch", "numpy", "robosuite", "mujoco", "libero")
_PACKAGE_KEYS = frozenset(
    {
        "module_name",
        "distribution_name",
        "version",
        "version_source",
        "module_dunder_version",
        "distribution_version",
        "module_file",
        "package_root",
        "distribution_root",
        "native_runtime_files",
    }
)
_ENVIRONMENT_IDENTITY_KEYS = frozenset(
    {
        "value",
        "availability",
        "sources",
        "checked_environment_variables",
    }
)


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("simulator runtime identity must be finite JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while it was hashed: {resolved}")
    if after.st_size <= 0:
        raise ValueError(f"{label} must not be empty: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(after.st_size),
    }


def _source_file_paths(source_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root)
        if _EXCLUDED_TREE_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink():
            if path.suffix.lower() in _SOURCE_SUFFIXES:
                raise ValueError(f"LIBERO source identity rejects symlinks: {path}")
            continue
        if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES:
            paths.append(path)
    return tuple(sorted(paths, key=lambda item: item.relative_to(source_root).as_posix()))


def _source_tree_ledger(source_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(source_root).as_posix(),
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        }
        for path in _source_file_paths(source_root)
        for identity in (
            _stable_file_identity(path, label=f"LIBERO source {path.name}"),
        )
    ]


def _libero_source_tree_identity(libero_root: str | Path) -> dict[str, Any]:
    root = Path(libero_root).expanduser().resolve(strict=True)
    source_root = (root / "libero" / "libero").resolve(strict=True)
    try:
        source_root.relative_to(root)
    except ValueError as error:
        raise ValueError("LIBERO source root escapes configured libero_root") from error
    if not source_root.is_dir():
        raise ValueError(f"LIBERO source root is not a directory: {source_root}")

    # The tree is small once meshes/textures are excluded.  Two complete passes
    # close the add/remove/replace race across the directory-wide snapshot.
    before = _source_tree_ledger(source_root)
    after = _source_tree_ledger(source_root)
    if before != after:
        raise RuntimeError("LIBERO source tree changed while it was hashed")
    if not before or not any(item["relative_path"].endswith(".py") for item in before):
        raise ValueError("LIBERO source tree contains no Python source files")
    return {
        "root": str(source_root),
        "coverage": "source_and_physics_configuration_files",
        "included_suffixes": list(_SOURCE_SUFFIXES),
        "excluded_directory_names": sorted(_EXCLUDED_TREE_PARTS),
        "file_count": len(before),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in before),
        "manifest_sha256": _canonical_sha256(before),
    }


def _module_file(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"required runtime module is unavailable: {module_name}")
    return Path(spec.origin).expanduser().resolve(strict=True)


def _distribution_version(distribution_name: str) -> tuple[str | None, str]:
    try:
        return (
            str(importlib.metadata.version(distribution_name)),
            f"importlib.metadata.version:{distribution_name}",
        )
    except importlib.metadata.PackageNotFoundError:
        return None, f"unavailable:no_distribution_metadata:{distribution_name}"


def _module_dunder_version(module_name: str) -> tuple[str | None, str]:
    module = importlib.import_module(module_name)
    value = getattr(module, "__version__", None)
    if value is None:
        return None, f"unavailable:{module_name}.__version__"
    return str(value), f"module_attribute:{module_name}.__version__"


def _package_identity(
    module_name: str,
    distribution_name: str,
    *,
    native_module_names: tuple[str, ...] = (),
    extra_native_files: tuple[Path, ...] = (),
) -> dict[str, Any]:
    module_path = _module_file(module_name)
    module_version, module_version_source = _module_dunder_version(module_name)
    distribution_version, distribution_version_source = _distribution_version(
        distribution_name
    )
    if module_version is not None:
        version, version_source = module_version, module_version_source
    else:
        version, version_source = distribution_version, distribution_version_source
    try:
        distribution_root = str(
            Path(importlib.metadata.distribution(distribution_name).locate_file(""))
            .expanduser()
            .resolve(strict=True)
        )
    except importlib.metadata.PackageNotFoundError:
        distribution_root = None

    native_files = [
        _stable_file_identity(
            _module_file(native_name), label=f"{native_name} native runtime"
        )
        for native_name in native_module_names
    ]
    native_files.extend(
        _stable_file_identity(path, label=f"{module_name} native runtime")
        for path in extra_native_files
    )
    native_files.sort(key=lambda item: item["path"])
    return {
        "module_name": module_name,
        "distribution_name": distribution_name,
        "version": version,
        "version_source": version_source,
        "module_dunder_version": {
            "value": module_version,
            "source": module_version_source,
        },
        "distribution_version": {
            "value": distribution_version,
            "source": distribution_version_source,
        },
        "module_file": _stable_file_identity(
            module_path, label=f"{module_name} module"
        ),
        "package_root": str(module_path.parent),
        "distribution_root": distribution_root,
        "native_runtime_files": native_files,
    }


def _libero_package_identity(source_tree: Mapping[str, Any]) -> dict[str, Any]:
    init_path = Path(str(source_tree["root"])) / "__init__.py"
    # LIBERO is a checked-out source tree rather than an installed distribution
    # in the formal environment.  Do not import it here because import-time
    # config creation is stateful; represent the missing version explicitly.
    distribution_version, distribution_source = _distribution_version("libero")
    return {
        "module_name": "libero",
        "distribution_name": "libero",
        "version": distribution_version,
        "version_source": distribution_source,
        "module_dunder_version": {
            "value": None,
            "source": "unavailable:not_imported_and_no_literal_version_contract",
        },
        "distribution_version": {
            "value": distribution_version,
            "source": distribution_source,
        },
        "module_file": _stable_file_identity(init_path, label="LIBERO module"),
        "package_root": str(init_path.parent.resolve(strict=True)),
        "distribution_root": None,
        "native_runtime_files": [],
    }


def _consistent_environment_identity(
    environ: Mapping[str, str],
    keys: tuple[str, ...],
    *,
    label: str,
    digest: bool,
) -> dict[str, Any]:
    present = [(key, str(environ[key]).strip()) for key in keys if environ.get(key)]
    if not present:
        return {
            "value": None,
            "availability": "unavailable",
            "sources": [],
            "checked_environment_variables": list(keys),
        }
    values = {value for _, value in present}
    if len(values) != 1:
        raise ValueError(f"conflicting {label} environment identities: {present}")
    value = next(iter(values))
    if digest and _OCI_DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an OCI sha256:<64 hex> digest")
    return {
        "value": value,
        "availability": "available",
        "sources": [f"environment:{key}" for key, _ in present],
        "checked_environment_variables": list(keys),
    }


def capture_libero_simulator_runtime_identity(
    libero_root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the external simulator runtime as a self-hashed JSON mapping."""

    environment = os.environ if environ is None else environ
    source_tree = _libero_source_tree_identity(libero_root)

    torch_path = _module_file("torch")
    torch_native_extra = torch_path.parent / "lib" / "libtorch_python.so"
    mujoco_path = _module_file("mujoco")
    mujoco_libraries = tuple(sorted(mujoco_path.parent.glob("libmujoco.so.*")))
    if len(mujoco_libraries) != 1:
        raise RuntimeError(
            "expected exactly one MuJoCo core shared library, found "
            f"{[str(path) for path in mujoco_libraries]}"
        )

    torch_package = _package_identity(
        "torch",
        "torch",
        native_module_names=("torch._C",),
        extra_native_files=(torch_native_extra,),
    )
    torch_module = importlib.import_module("torch")
    torch_package["runtime_build"] = {
        "cuda": None if torch_module.version.cuda is None else str(torch_module.version.cuda),
        "cudnn": (
            None
            if torch_module.backends.cudnn.version() is None
            else str(torch_module.backends.cudnn.version())
        ),
        "cxx11_abi": bool(torch_module._C._GLIBCXX_USE_CXX11_ABI),
    }

    packages = {
        "torch": torch_package,
        "numpy": _package_identity(
            "numpy", "numpy", native_module_names=("numpy._core._multiarray_umath",)
        ),
        "robosuite": _package_identity("robosuite", "robosuite"),
        "mujoco": _package_identity(
            "mujoco",
            "mujoco",
            native_module_names=("mujoco._functions", "mujoco._structs"),
            extra_native_files=mujoco_libraries,
        ),
        "libero": _libero_package_identity(source_tree),
    }
    executable = _stable_file_identity(sys.executable, label="Python executable")
    libc_name, libc_version = platform.libc_ver()
    payload = {
        "schema_version": LIBERO_SIMULATOR_IDENTITY_SCHEMA_VERSION,
        "kind": LIBERO_SIMULATOR_IDENTITY_KIND,
        "python": {
            "version": platform.python_version(),
            "version_source": "platform.python_version",
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "prefix": str(Path(sys.prefix).expanduser().resolve(strict=True)),
            "base_prefix": str(
                Path(sys.base_prefix).expanduser().resolve(strict=True)
            ),
            "executable": executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": {"name": libc_name or None, "version": libc_version or None},
        },
        "packages": packages,
        "libero_source_tree": source_tree,
        "runtime_environment": {
            "renderer_policy": _RENDERER_POLICY,
            "variables": {
                key: (str(environment[key]) if key in environment else None)
                for key in _RUNTIME_ENVIRONMENT_KEYS
            },
            "container_image_digest": _consistent_environment_identity(
                environment,
                _IMAGE_DIGEST_ENV_KEYS,
                label="container image digest",
                digest=True,
            ),
            "jihe_environment_identity": _consistent_environment_identity(
                environment,
                _JIHE_ENVIRONMENT_ID_KEYS,
                label="JiHe environment",
                digest=False,
            ),
        },
    }
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _schema_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _schema_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{field} schema mismatch: missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _schema_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _schema_optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _schema_nonempty_string(value, field=field)


def _schema_absolute_path(value: Any, *, field: str) -> str:
    result = _schema_nonempty_string(value, field=field)
    if not Path(result).is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return result


def _schema_optional_absolute_path(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _schema_absolute_path(value, field=field)


def _schema_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _schema_positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_file_identity_schema(value: Any, *, field: str) -> dict[str, Any]:
    identity = _schema_mapping(value, field=field)
    _schema_exact_keys(
        identity,
        frozenset({"path", "sha256", "size_bytes"}),
        field=field,
    )
    _schema_absolute_path(identity["path"], field=f"{field}.path")
    _schema_sha256(identity["sha256"], field=f"{field}.sha256")
    _schema_positive_integer(identity["size_bytes"], field=f"{field}.size_bytes")
    return identity


def _validate_version_record_schema(
    value: Any,
    *,
    field: str,
) -> tuple[str | None, str]:
    record = _schema_mapping(value, field=field)
    _schema_exact_keys(record, frozenset({"value", "source"}), field=field)
    version = _schema_optional_string(record["value"], field=f"{field}.value")
    source = _schema_nonempty_string(record["source"], field=f"{field}.source")
    return version, source


def _validate_package_identity_schema(
    value: Any,
    *,
    package_name: str,
) -> dict[str, Any]:
    field = f"packages.{package_name}"
    package = _schema_mapping(value, field=field)
    expected_keys = set(_PACKAGE_KEYS)
    if package_name == "torch":
        expected_keys.add("runtime_build")
    _schema_exact_keys(package, expected_keys, field=field)

    if package["module_name"] != package_name:
        raise ValueError(f"{field}.module_name must be {package_name!r}")
    if package["distribution_name"] != package_name:
        raise ValueError(f"{field}.distribution_name must be {package_name!r}")

    version = _schema_optional_string(package["version"], field=f"{field}.version")
    version_source = _schema_nonempty_string(
        package["version_source"], field=f"{field}.version_source"
    )
    module_version, module_source = _validate_version_record_schema(
        package["module_dunder_version"],
        field=f"{field}.module_dunder_version",
    )
    distribution_version, distribution_source = _validate_version_record_schema(
        package["distribution_version"],
        field=f"{field}.distribution_version",
    )

    if package_name == "libero":
        expected_module_source = (
            "unavailable:not_imported_and_no_literal_version_contract"
        )
        if module_version is not None or module_source != expected_module_source:
            raise ValueError(
                "packages.libero.module_dunder_version must explicitly record "
                "the unavailable non-imported version"
            )
    else:
        expected_module_source = (
            f"module_attribute:{package_name}.__version__"
            if module_version is not None
            else f"unavailable:{package_name}.__version__"
        )
        if module_source != expected_module_source:
            raise ValueError(f"{field}.module_dunder_version.source is invalid")

    expected_distribution_source = (
        f"importlib.metadata.version:{package_name}"
        if distribution_version is not None
        else f"unavailable:no_distribution_metadata:{package_name}"
    )
    if distribution_source != expected_distribution_source:
        raise ValueError(f"{field}.distribution_version.source is invalid")

    selected_version = (
        module_version if module_version is not None else distribution_version
    )
    selected_source = (
        module_source if module_version is not None else distribution_source
    )
    if version != selected_version or version_source != selected_source:
        raise ValueError(f"{field} selected version/source is inconsistent")

    module_file = _validate_file_identity_schema(
        package["module_file"], field=f"{field}.module_file"
    )
    package_root = _schema_absolute_path(
        package["package_root"], field=f"{field}.package_root"
    )
    if Path(module_file["path"]).parent != Path(package_root):
        raise ValueError(f"{field}.package_root must contain module_file")
    _schema_optional_absolute_path(
        package["distribution_root"], field=f"{field}.distribution_root"
    )

    native_files = package["native_runtime_files"]
    if not isinstance(native_files, list):
        raise ValueError(f"{field}.native_runtime_files must be a list")
    validated_native_files = [
        _validate_file_identity_schema(
            native_file,
            field=f"{field}.native_runtime_files[{index}]",
        )
        for index, native_file in enumerate(native_files)
    ]
    native_paths = [item["path"] for item in validated_native_files]
    if native_paths != sorted(native_paths) or len(native_paths) != len(
        set(native_paths)
    ):
        raise ValueError(
            f"{field}.native_runtime_files must have unique paths in sorted order"
        )
    expected_native_counts = {
        "torch": 2,
        "numpy": 1,
        "robosuite": 0,
        "mujoco": 3,
        "libero": 0,
    }
    if len(validated_native_files) != expected_native_counts[package_name]:
        raise ValueError(
            f"{field}.native_runtime_files must contain exactly "
            f"{expected_native_counts[package_name]} entries"
        )

    if package_name == "torch":
        runtime_build = _schema_mapping(
            package["runtime_build"], field="packages.torch.runtime_build"
        )
        _schema_exact_keys(
            runtime_build,
            frozenset({"cuda", "cudnn", "cxx11_abi"}),
            field="packages.torch.runtime_build",
        )
        _schema_optional_string(
            runtime_build["cuda"], field="packages.torch.runtime_build.cuda"
        )
        _schema_optional_string(
            runtime_build["cudnn"], field="packages.torch.runtime_build.cudnn"
        )
        if not isinstance(runtime_build["cxx11_abi"], bool):
            raise ValueError(
                "packages.torch.runtime_build.cxx11_abi must be boolean"
            )
    return package


def _validate_environment_identity_schema(
    value: Any,
    *,
    field: str,
    checked_keys: tuple[str, ...],
    require_oci_digest: bool,
) -> None:
    identity = _schema_mapping(value, field=field)
    _schema_exact_keys(identity, _ENVIRONMENT_IDENTITY_KEYS, field=field)
    checked = identity["checked_environment_variables"]
    if checked != list(checked_keys):
        raise ValueError(
            f"{field}.checked_environment_variables does not match the schema"
        )
    sources = identity["sources"]
    if not isinstance(sources, list):
        raise ValueError(f"{field}.sources must be a list")
    allowed_sources = [f"environment:{key}" for key in checked_keys]
    if (
        any(not isinstance(source, str) for source in sources)
        or len(sources) != len(set(sources))
        or sources != [source for source in allowed_sources if source in sources]
    ):
        raise ValueError(
            f"{field}.sources must be a unique ordered subset of checked variables"
        )

    availability = identity["availability"]
    if availability == "unavailable":
        if identity["value"] is not None or sources:
            raise ValueError(
                f"{field} unavailable identity must have null value and no sources"
            )
    elif availability == "available":
        identity_value = _schema_nonempty_string(
            identity["value"], field=f"{field}.value"
        )
        if not sources:
            raise ValueError(f"{field} available identity must record a source")
        if require_oci_digest and _OCI_DIGEST_RE.fullmatch(identity_value) is None:
            raise ValueError(f"{field}.value must be an OCI SHA256 digest")
    else:
        raise ValueError(f"{field}.availability must be available or unavailable")


def _validate_deep_identity_schema(normalized: dict[str, Any]) -> None:
    python = _schema_mapping(normalized["python"], field="python")
    _schema_exact_keys(
        python,
        frozenset(
            {
                "version",
                "version_source",
                "implementation",
                "cache_tag",
                "prefix",
                "base_prefix",
                "executable",
            }
        ),
        field="python",
    )
    _schema_nonempty_string(python["version"], field="python.version")
    if python["version_source"] != "platform.python_version":
        raise ValueError("python.version_source must be platform.python_version")
    _schema_nonempty_string(python["implementation"], field="python.implementation")
    _schema_nonempty_string(python["cache_tag"], field="python.cache_tag")
    _schema_absolute_path(python["prefix"], field="python.prefix")
    _schema_absolute_path(python["base_prefix"], field="python.base_prefix")
    _validate_file_identity_schema(python["executable"], field="python.executable")

    platform_identity = _schema_mapping(normalized["platform"], field="platform")
    _schema_exact_keys(
        platform_identity,
        frozenset({"system", "release", "machine", "libc"}),
        field="platform",
    )
    for key in ("system", "release", "machine"):
        _schema_nonempty_string(
            platform_identity[key], field=f"platform.{key}"
        )
    libc = _schema_mapping(platform_identity["libc"], field="platform.libc")
    _schema_exact_keys(libc, frozenset({"name", "version"}), field="platform.libc")
    _schema_optional_string(libc["name"], field="platform.libc.name")
    _schema_optional_string(libc["version"], field="platform.libc.version")

    source_tree = _schema_mapping(
        normalized["libero_source_tree"], field="libero_source_tree"
    )
    _schema_exact_keys(
        source_tree,
        frozenset(
            {
                "root",
                "coverage",
                "included_suffixes",
                "excluded_directory_names",
                "file_count",
                "total_size_bytes",
                "manifest_sha256",
            }
        ),
        field="libero_source_tree",
    )
    source_root = _schema_absolute_path(
        source_tree["root"], field="libero_source_tree.root"
    )
    if tuple(Path(source_root).parts[-2:]) != ("libero", "libero"):
        raise ValueError("libero_source_tree.root must end in libero/libero")
    if source_tree["coverage"] != "source_and_physics_configuration_files":
        raise ValueError("libero_source_tree.coverage is unsupported")
    if source_tree["included_suffixes"] != list(_SOURCE_SUFFIXES):
        raise ValueError("libero_source_tree.included_suffixes mismatch")
    if source_tree["excluded_directory_names"] != sorted(_EXCLUDED_TREE_PARTS):
        raise ValueError("libero_source_tree.excluded_directory_names mismatch")
    _schema_positive_integer(
        source_tree["file_count"], field="libero_source_tree.file_count"
    )
    _schema_positive_integer(
        source_tree["total_size_bytes"],
        field="libero_source_tree.total_size_bytes",
    )
    _schema_sha256(
        source_tree["manifest_sha256"],
        field="libero_source_tree.manifest_sha256",
    )

    packages = _schema_mapping(normalized["packages"], field="packages")
    _schema_exact_keys(packages, frozenset(_PACKAGE_NAMES), field="packages")
    validated_packages = {
        name: _validate_package_identity_schema(packages[name], package_name=name)
        for name in _PACKAGE_NAMES
    }
    libero_package = validated_packages["libero"]
    if libero_package["package_root"] != source_root:
        raise ValueError("packages.libero.package_root must equal LIBERO source root")
    expected_libero_module = str(Path(source_root) / "__init__.py")
    if libero_package["module_file"]["path"] != expected_libero_module:
        raise ValueError(
            "packages.libero.module_file must be LIBERO source __init__.py"
        )
    if libero_package["distribution_root"] is not None:
        raise ValueError("packages.libero.distribution_root must be null")

    runtime = _schema_mapping(
        normalized["runtime_environment"], field="runtime_environment"
    )
    _schema_exact_keys(
        runtime,
        frozenset(
            {
                "renderer_policy",
                "variables",
                "container_image_digest",
                "jihe_environment_identity",
            }
        ),
        field="runtime_environment",
    )
    if runtime["renderer_policy"] != _RENDERER_POLICY:
        raise ValueError("runtime_environment.renderer_policy is unsupported")
    variables = _schema_mapping(
        runtime["variables"], field="runtime_environment.variables"
    )
    _schema_exact_keys(
        variables,
        frozenset(_RUNTIME_ENVIRONMENT_KEYS),
        field="runtime_environment.variables",
    )
    for key, variable in variables.items():
        if variable is not None and not isinstance(variable, str):
            raise ValueError(
                f"runtime_environment.variables.{key} must be a string or null"
            )
    configured_libero_root = variables["FASTWAM_LIBERO_ROOT"]
    if configured_libero_root is not None:
        _schema_absolute_path(
            configured_libero_root,
            field="runtime_environment.variables.FASTWAM_LIBERO_ROOT",
        )
        if Path(configured_libero_root) / "libero" / "libero" != Path(
            source_root
        ):
            raise ValueError(
                "runtime_environment FASTWAM_LIBERO_ROOT disagrees with source root"
            )
    _validate_environment_identity_schema(
        runtime["container_image_digest"],
        field="runtime_environment.container_image_digest",
        checked_keys=_IMAGE_DIGEST_ENV_KEYS,
        require_oci_digest=True,
    )
    _validate_environment_identity_schema(
        runtime["jihe_environment_identity"],
        field="runtime_environment.jihe_environment_identity",
        checked_keys=_JIHE_ENVIRONMENT_ID_KEYS,
        require_oci_digest=False,
    )


def _validated_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("simulator runtime identity must be a mapping")
    try:
        normalized = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("simulator runtime identity must be finite JSON data") from error
    expected_keys = {
        "schema_version",
        "kind",
        "python",
        "platform",
        "packages",
        "libero_source_tree",
        "runtime_environment",
        "identity_sha256",
    }
    if set(normalized) != expected_keys:
        raise ValueError("simulator runtime identity top-level schema mismatch")
    if (
        normalized["schema_version"] != LIBERO_SIMULATOR_IDENTITY_SCHEMA_VERSION
        or normalized["kind"] != LIBERO_SIMULATOR_IDENTITY_KIND
    ):
        raise ValueError("unsupported LIBERO simulator runtime identity")
    source_tree = normalized.get("libero_source_tree")
    if not isinstance(source_tree, dict):
        raise ValueError("simulator runtime identity libero_source_tree must be a mapping")
    root = source_tree.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("simulator runtime identity LIBERO source root must be absolute")
    recorded_sha = normalized.get("identity_sha256")
    if not isinstance(recorded_sha, str) or re.fullmatch(r"[0-9a-f]{64}", recorded_sha) is None:
        raise ValueError("simulator runtime identity identity_sha256 is invalid")
    unhashed = dict(normalized)
    unhashed.pop("identity_sha256")
    if _canonical_sha256(unhashed) != recorded_sha:
        raise ValueError("simulator runtime identity self-SHA256 mismatch")
    _validate_deep_identity_schema(normalized)
    return normalized


def validate_libero_simulator_runtime_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate schema/self-hash without touching the referenced files."""

    return _validated_identity_payload(value)


def verify_libero_simulator_runtime_identity(
    expected: Mapping[str, Any],
    *,
    libero_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Physically recapture a runtime and require exact identity equality."""

    normalized = _validated_identity_payload(expected)
    expected_source_root = Path(normalized["libero_source_tree"]["root"])
    configured_root = (
        expected_source_root.parents[1]
        if libero_root is None
        else Path(libero_root).expanduser().resolve(strict=True)
    )
    if (configured_root / "libero" / "libero").resolve(strict=True) != expected_source_root:
        raise ValueError("configured LIBERO root disagrees with simulator runtime identity")
    observed = capture_libero_simulator_runtime_identity(
        configured_root, environ=environ
    )
    if observed != normalized:
        raise ValueError(
            "LIBERO simulator runtime identity drifted: "
            f"expected={normalized['identity_sha256']}, "
            f"actual={observed['identity_sha256']}"
        )
    return normalized


def verify_loaded_libero_module_location(
    identity: Mapping[str, Any], module_file: str | Path
) -> None:
    """Require an imported LIBERO module to come from the frozen source tree."""

    normalized = _validated_identity_payload(identity)
    source_root = Path(normalized["libero_source_tree"]["root"]).resolve(strict=True)
    loaded = Path(module_file).expanduser().resolve(strict=True)
    try:
        loaded.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"loaded LIBERO module is outside frozen source tree: {loaded}"
        ) from error


__all__ = [
    "LIBERO_SIMULATOR_IDENTITY_KIND",
    "LIBERO_SIMULATOR_IDENTITY_SCHEMA_VERSION",
    "capture_libero_simulator_runtime_identity",
    "validate_libero_simulator_runtime_identity",
    "verify_libero_simulator_runtime_identity",
    "verify_loaded_libero_module_location",
]
