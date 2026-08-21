import importlib.util
from pathlib import Path

import pytest


def _load_docs_build_module():
    script = Path(__file__).parents[1] / "docs" / "build.py"
    spec = importlib.util.spec_from_file_location("biolgca_docs_build", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_generated_removes_stale_autosummary_and_html(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "_build" / "html"
    autosummary_dir = source_dir / "_autosummary"
    autosummary_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (autosummary_dir / "removed_api.rst").write_text("stale", encoding="utf-8")
    (output_dir / "index.html").write_text("stale", encoding="utf-8")

    build = _load_docs_build_module()
    build.clean_generated(source_dir, output_dir)

    assert not autosummary_dir.exists()
    assert not output_dir.exists()


def test_clean_generated_propagates_removal_failures(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    autosummary_dir = source_dir / "_autosummary"
    autosummary_dir.mkdir(parents=True)
    build = _load_docs_build_module()

    def fail_if_errors_are_not_ignored(path, *, ignore_errors=False):
        if not ignore_errors:
            raise PermissionError(path)

    monkeypatch.setattr(build.shutil, "rmtree", fail_if_errors_are_not_ignored)

    with pytest.raises(PermissionError):
        build.clean_generated(source_dir, tmp_path / "html")
