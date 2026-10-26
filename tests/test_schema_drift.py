import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "archive_data_schemas.py").resolve()


def run_audit(data_dir, baseline, *args):
    env = os.environ.copy()
    env["EDEN_DATA_DIR"] = str(data_dir)
    env["EDEN_BASELINE_PATH"] = str(baseline)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=SCRIPT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def snapshot(root):
    if not root.exists():
        return None
    return {
        path.relative_to(root).as_posix(): (
            "dir" if path.is_dir() else path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
    }


def generate(data_dir, baseline):
    result = run_audit(data_dir, baseline, "--generate")
    assert result.returncode == 0, result.stdout + result.stderr
    assert baseline.is_file()
    return json.loads(baseline.read_text(encoding="utf-8"))


def test_check_is_read_only_when_baseline_is_missing(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    baseline = tmp_path / "docs" / "baseline.json"
    before = snapshot(tmp_path)
    result = run_audit(data, baseline, "--check")
    after = snapshot(tmp_path)
    assert result.returncode != 0
    assert "absente" in result.stderr
    assert before == after


def test_jsonl_keys_are_flat_and_types_are_normalized(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.jsonl").write_text(
        '{"id": 1, "name": "A", "nullable": null}\n'
        '{"id": 2, "name": "B", "nullable": "x"}\n',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    document = generate(data, baseline)
    schema = document["schemas"]["sample.jsonl"]
    assert schema["columns_or_keys"] == {
        "id": "integer", "name": "string", "nullable": "string"
    }
    before = snapshot(tmp_path)
    result = run_audit(data, baseline, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(tmp_path) == before


def test_empty_catalog_is_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    result = run_audit(data, tmp_path / "baseline.json", "--check")
    assert result.returncode == 2
    assert "Aucun fichier" in result.stderr


@pytest.mark.parametrize("content", ["[]\n", '"scalar"\n', "not-json\n"])
def test_invalid_jsonl_is_rejected(tmp_path, content):
    data = tmp_path / "data"
    data.mkdir()
    (data / "bad.jsonl").write_text(content, encoding="utf-8")
    result = run_audit(data, tmp_path / "baseline.json", "--generate")
    assert result.returncode == 2
    assert not (tmp_path / "baseline.json").exists()


def test_jsonl_conflicting_types_are_marked_mixed_and_null_is_tolerated(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "mixed.jsonl").write_text(
        '{"x": null, "y": 1}\n{"x": "text", "y": 2.5}\n',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    schema = generate(data, baseline)["schemas"]["mixed.jsonl"]
    assert schema["columns_or_keys"] == {"x": "string", "y": "mixed"}


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"version": 1},
        {"version": 1, "schemas": []},
        {"version": 1, "schemas": {"x.jsonl": {}}},
        {"version": 1, "schemas": {"x.jsonl": {
            "format": "jsonl", "columns_or_keys": {"id": "integer"},
            "structural_fingerprint_sha256": "not-a-hash"
        }}},
    ],
)
def test_malformed_baseline_structures_are_rejected(tmp_path, document):
    data = tmp_path / "data"
    data.mkdir()
    (data / "x.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(document), encoding="utf-8")
    result = run_audit(data, baseline, "--check")
    assert result.returncode == 2
    assert "Baseline" in result.stderr or "baseline" in result.stderr


def test_new_file_and_column_drift_are_reported(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "matches.jsonl"
    source.write_text('{"id": 1, "home": "A"}\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    generate(data, baseline)
    source.write_text('{"id": 1, "away": "B"}\n', encoding="utf-8")
    (data / "new.jsonl").write_text('{"new": true}\n', encoding="utf-8")
    result = run_audit(data, baseline, "--check")
    assert result.returncode == 1
    assert "colonne/clé approuvée manquante: home" in result.stderr
    assert "nouvelle colonne/clé non approuvée: away" in result.stderr
    assert "Nouveau fichier non approuvé: new.jsonl" in result.stderr


def test_csv_type_drift_is_detected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    csv = data / "matches.csv"
    csv.write_text("id,team\n1,A\n2,B\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    generate(data, baseline)
    csv.write_text("id,team\none,A\n2,B\n", encoding="utf-8")
    result = run_audit(data, baseline, "--check")
    assert result.returncode == 1
    assert "type de id" in result.stderr


def test_missing_approved_file_is_reported(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "one.jsonl"
    source.write_text('{"id": 1}\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    generate(data, baseline)
    source.unlink()
    (data / "other.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
    result = run_audit(data, baseline, "--check")
    assert result.returncode == 1
    assert "Fichier approuvé manquant: one.jsonl" in result.stderr


def test_empty_csv_is_rejected_without_baseline_write(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "empty.csv").write_text("", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    result = run_audit(data, baseline, "--generate")
    assert result.returncode == 2
    assert not baseline.exists()


def test_check_rejects_nonexistent_data_directory(tmp_path):
    result = run_audit(tmp_path / "missing", tmp_path / "baseline.json", "--check")
    assert result.returncode == 2
    assert "absent ou invalide" in result.stderr
