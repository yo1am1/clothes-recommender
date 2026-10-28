from pathlib import Path

import pytest

from scripts import compare_rankers


def test_missing_optional_dependency_precedes_cli_input_hashing(tmp_path, monkeypatch, capsys):
    def missing():
        raise ValueError("missing optional LightGBM")

    def unexpected(*args, **kwargs):
        pytest.fail("Input files were opened before optional dependency validation")

    monkeypatch.setattr(compare_rankers, "_lightgbm", missing)
    monkeypatch.setattr(Path, "open", unexpected)
    with pytest.raises(SystemExit):
        compare_rankers.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--holdout-end",
                "2020-09-01",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
    assert "missing optional LightGBM" in capsys.readouterr().err
