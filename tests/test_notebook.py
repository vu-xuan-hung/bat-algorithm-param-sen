from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.run_notebook import execute_notebook

ROOT = Path(__file__).resolve().parents[1]


def write_notebook(path: Path, sources: list[str]) -> None:
    cells = [
        {
            "cell_type": "code",
            "id": f"cell-{i}",
            "metadata": {},
            "source": [source],
            "execution_count": 99,
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["STALE OUTPUT"]}],
        }
        for i, source in enumerate(sources)
    ]
    path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells})
    )


def test_runner_refuses_output_overwrite_before_executing(tmp_path: Path) -> None:
    source, target = tmp_path / "source.ipynb", tmp_path / "target.ipynb"
    marker = tmp_path / "marker"
    write_notebook(source, [f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    target.write_text("precious output")
    with pytest.raises(FileExistsError):
        execute_notebook(source, target)
    assert target.read_text() == "precious output"
    assert not marker.exists()


def test_runner_records_failure_and_clears_unexecuted_stale_outputs(tmp_path: Path) -> None:
    source, target = tmp_path / "source.ipynb", tmp_path / "target.ipynb"
    write_notebook(source, ["print('before'); raise ValueError('real error')", "print('not run')"])
    with pytest.raises(RuntimeError, match="Notebook failed"):
        execute_notebook(source, target)
    result = json.loads(target.read_text())
    assert result["metadata"]["execution_verification"]["status"] == "failed"
    assert result["cells"][0]["outputs"][-1]["ename"] == "ValueError"
    assert result["cells"][1]["execution_count"] is None
    assert result["cells"][1]["outputs"] == []


@pytest.mark.integration
def test_notebook_runs_on_user_csv_with_labels_and_one_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "user data.csv"
    dataset.write_text("value,truth\n1,left\n2,left\n3,left\n8,right\n9,right\n10,right\n")
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("BA_NOTEBOOK_DATASET", str(dataset))
    monkeypatch.setenv("BA_NOTEBOOK_CLUSTERS", "2")
    monkeypatch.setenv("BA_NOTEBOOK_LABEL_COLUMN", "truth")
    monkeypatch.setenv("BA_NOTEBOOK_OUTPUT_ROOT", str(tmp_path / "results"))
    result = execute_notebook(
        ROOT / "notebooks/parameter_sensitivity.ipynb", tmp_path / "run.ipynb"
    )
    assert result["metadata"]["execution_verification"]["status"] == "passed"
    code = [c for c in result["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is not None for c in code)
    images = [o for c in code for o in c["outputs"] if "image/png" in o.get("data", {})]
    assert len(images) == 11
    session = next((tmp_path / "results").iterdir())
    evidence = json.loads((session / "notebook_evidence.json").read_text())
    assert evidence["ofat_records"] == 28
    assert evidence["dataset_path"] == str(dataset)
    assert evidence["reference_recomputation_passed"]
    manifest = json.loads((session / "ofat/manifest.json").read_text())
    assert manifest["datasets"]["custom"]["n_features"] == 1
    assert manifest["spec"]["seeds"] == [0, 1]
    assert len(list(session.rglob("*.png"))) == 11
