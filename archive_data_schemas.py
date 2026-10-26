#!/usr/bin/env python3
"""Read-only structural audit for archived CSV and JSONL data.

The inferred types describe the currently observed files; they are not a substitute
for an upstream data contract. Only --generate writes the approved baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASELINE_VERSION = 1
ALLOWED_FORMATS = {"csv", "jsonl"}


class SchemaAuditError(RuntimeError):
    """Raised when the data catalog or baseline cannot be audited safely."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_pandas_dtype(dtype: Any) -> str:
    name = str(dtype).lower()
    if name in {"object", "string", "str"}:
        return "string"
    if name.startswith(("int", "uint")):
        return "integer"
    if name.startswith("float"):
        return "float"
    if name in {"bool", "boolean"}:
        return "boolean"
    if name.startswith("datetime"):
        return "datetime"
    return name


def json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def observed_jsonl_types(path: Path) -> dict[str, str]:
    types_by_key: dict[str, set[str]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SchemaAuditError(
                        f"{path}: JSON invalide à la ligne {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise SchemaAuditError(
                        f"{path}: la ligne {line_number} doit contenir un objet JSON."
                    )
                for key, value in record.items():
                    types_by_key.setdefault(str(key), set()).add(json_value_type(value))
    except OSError as exc:
        raise SchemaAuditError(f"Lecture impossible de {path}: {exc}") from exc

    result: dict[str, str] = {}
    for key in sorted(types_by_key):
        observed = types_by_key[key]
        non_null = observed - {"null"}
        if not non_null:
            result[key] = "null"
        elif len(non_null) == 1:
            result[key] = next(iter(non_null))
        else:
            result[key] = "mixed"
    return result


def schema_for(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            # Read only the header first, so empty files and headerless files fail clearly.
            header = pd.read_csv(path, nrows=0)
            if len(header.columns) == 0:
                raise SchemaAuditError(f"{path}: CSV sans colonnes.")
            frame = pd.read_csv(path, low_memory=False)
        except SchemaAuditError:
            raise
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise SchemaAuditError(f"CSV illisible ou invalide {path}: {exc}") from exc
        columns = {str(name): normalize_pandas_dtype(dtype) for name, dtype in frame.dtypes.items()}
        fmt = "csv"
    elif suffix == ".jsonl":
        columns = observed_jsonl_types(path)
        if not columns:
            raise SchemaAuditError(f"{path}: JSONL vide ou sans clés observées.")
        fmt = "jsonl"
    else:
        raise SchemaAuditError(f"Format non pris en charge: {path}")
    structure = {"format": fmt, "columns_or_keys": columns}
    return {
        **structure,
        "structural_fingerprint_sha256": fingerprint(structure),
    }


def scan_catalog(data_dir: Path) -> dict[str, dict[str, Any]]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise SchemaAuditError(f"Répertoire de données absent ou invalide: {data_dir}")

    paths: list[Path] = []
    try:
        for root, dirs, files in os.walk(data_dir, onerror=lambda exc: (_ for _ in ()).throw(exc)):
            dirs.sort()
            for filename in sorted(files):
                path = Path(root) / filename
                if path.suffix.lower() in {".csv", ".jsonl"}:
                    paths.append(path)
    except OSError as exc:
        raise SchemaAuditError(f"Impossible de parcourir {data_dir}: {exc}") from exc

    if not paths:
        raise SchemaAuditError(f"Aucun fichier CSV ou JSONL trouvé dans {data_dir}.")

    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: item.relative_to(data_dir).as_posix()):
        relative = path.relative_to(data_dir).as_posix()
        catalog[relative] = schema_for(path)
    return catalog


def validate_baseline(value: Any, baseline_path: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or value.get("version") != BASELINE_VERSION:
        raise SchemaAuditError(
            f"{baseline_path}: racine invalide (version attendue: {BASELINE_VERSION})."
        )
    schemas = value.get("schemas")
    if not isinstance(schemas, dict) or not schemas:
        raise SchemaAuditError(f"{baseline_path}: champ 'schemas' absent, vide ou invalide.")

    for relative, spec in schemas.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise SchemaAuditError(f"{baseline_path}: chemin de fichier invalide: {relative!r}")
        if ".." in Path(relative).parts:
            raise SchemaAuditError(f"{baseline_path}: chemin sortant du catalogue: {relative!r}")
        if not isinstance(spec, dict):
            raise SchemaAuditError(f"{baseline_path}: schéma invalide pour {relative!r}.")
        if spec.get("format") not in ALLOWED_FORMATS:
            raise SchemaAuditError(f"{baseline_path}: format invalide pour {relative!r}.")
        columns = spec.get("columns_or_keys")
        if not isinstance(columns, dict) or not columns:
            raise SchemaAuditError(f"{baseline_path}: 'columns_or_keys' invalide pour {relative!r}.")
        if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in columns.items()):
            raise SchemaAuditError(f"{baseline_path}: types de colonnes invalides pour {relative!r}.")
        expected = spec.get("structural_fingerprint_sha256")
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise SchemaAuditError(f"{baseline_path}: empreinte SHA-256 invalide pour {relative!r}.")
        structure = {"format": spec["format"], "columns_or_keys": columns}
        if fingerprint(structure) != expected:
            raise SchemaAuditError(f"{baseline_path}: empreinte incohérente pour {relative!r}.")
    return schemas


def read_baseline(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SchemaAuditError(
            f"Baseline approuvée absente: {path}. Exécuter --generate uniquement après revue explicite."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaAuditError(f"Baseline illisible ou JSON invalide {path}: {exc}") from exc
    return validate_baseline(value, path)


def compare_catalog(
    approved: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    for path in sorted(approved.keys() - current.keys()):
        issues.append(f"Fichier approuvé manquant: {path}")
    for path in sorted(current.keys() - approved.keys()):
        issues.append(f"Nouveau fichier non approuvé: {path}")
    for path in sorted(approved.keys() & current.keys()):
        old, new = approved[path], current[path]
        if old["format"] != new["format"]:
            issues.append(f"{path}: format {old['format']} -> {new['format']}")
        old_columns, new_columns = old["columns_or_keys"], new["columns_or_keys"]
        for key in sorted(old_columns.keys() - new_columns.keys()):
            issues.append(f"{path}: colonne/clé approuvée manquante: {key}")
        for key in sorted(new_columns.keys() - old_columns.keys()):
            issues.append(f"{path}: nouvelle colonne/clé non approuvée: {key}")
        for key in sorted(old_columns.keys() & new_columns.keys()):
            if old_columns[key] != new_columns[key]:
                issues.append(f"{path}: type de {key}: {old_columns[key]} -> {new_columns[key]}")
        if old["structural_fingerprint_sha256"] != new["structural_fingerprint_sha256"] and not any(
            issue.startswith(f"{path}:") or issue.endswith(f": {path}") for issue in issues
        ):
            issues.append(f"{path}: empreinte structurelle modifiée.")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Compare sans écrire.")
    mode.add_argument("--generate", action="store_true", help="Génère une baseline (action explicite).")
    parser.add_argument("--data-dir", default=os.environ.get("EDEN_DATA_DIR", "data"))
    parser.add_argument(
        "--baseline",
        default=os.environ.get("EDEN_BASELINE_PATH", "docs/data_schema_baseline.json"),
    )
    args = parser.parse_args(argv)
    data_dir, baseline_path = Path(args.data_dir), Path(args.baseline)

    try:
        current = scan_catalog(data_dir)
        if args.generate:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            document = {"version": BASELINE_VERSION, "schemas": current}
            baseline_path.write_text(
                json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Baseline générée explicitement: {baseline_path} ({len(current)} fichier(s)).")
            return 0

        approved = read_baseline(baseline_path)
        issues = compare_catalog(approved, current)
        if issues:
            print("ÉCHEC — dérive de schéma détectée:", file=sys.stderr)
            for issue in issues:
                print(f"- {issue}", file=sys.stderr)
            return 1
        print(f"OK — {len(current)} fichier(s) conforme(s) à la baseline approuvée.")
        return 0
    except (SchemaAuditError, OSError) as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
