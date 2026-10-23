"""The command line as a researcher uses it with rules of her own (tutorial 6's project layout).

The project folder holds the model file and the module of the rules, ``model.json`` and
``hypoxia_rules.py``; she runs ``biolgca run model.json --plugins hypoxia_rules``.
"""

import importlib
import json
import sys
import textwrap

import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, save_model_spec
from lgca.pipeline import InteractionPipelineSpec

RULES = """
    from lgca.rules import interaction

    @interaction(kind="birth_death", families=("classical",), name="hypoxia_rules.crowding_death")
    def crowding_death(state, r_d=0.2):
        state.remove_cells(r_d * state.density / state.capacity)
"""
BROKEN = """
    from lgca.rules import interaction

    @interaction(kind="birth_death", families=("classical",), name="hypoxia_rules.crowding_death")
    def crowding_death(state, r_d=0.2):
        rates = {"r_d": r_d}
        state.remove_cells(rates["rd"])
"""


def _main(argv):
    return importlib.import_module("lgca.cli").main(argv)


@pytest.fixture
def project(tmp_path, request):
    """A folder with model.json, which names the rule of hypoxia_rules.py (not imported)."""
    folder = tmp_path / "project"
    folder.mkdir()
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(4, 4)), state=StateSpec(density=1.0),
                     time=TimeSpec(steps=2, seed=1), dynamics=InteractionPipelineSpec(operators=[]))
    data = json.loads(save_model_spec(spec, folder / "model.json").read_text(encoding="utf-8"))
    data["model"]["dynamics"]["operators"] = [{"name": "hypoxia_rules.crowding_death"}]
    (folder / "model.json").write_text(json.dumps(data), encoding="utf-8")
    request.addfinalizer(lambda: sys.modules.pop("hypoxia_rules", None))
    return folder


def _write_rules(folder, source):
    (folder / "hypoxia_rules.py").write_text(textwrap.dedent(source), encoding="utf-8")


@pytest.mark.parametrize("working_directory", ["project", "elsewhere"])
def test_rules_module_next_to_the_model_file_is_imported(project, tmp_path, monkeypatch, working_directory):
    _write_rules(project, RULES)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / working_directory)
    path_before = list(sys.path)
    assert _main(["run", str(project / "model.json"), "--plugins", "hypoxia_rules",
                  "--output", str(tmp_path / "run")]) == 0
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["steps"] == 2
    assert sys.path == path_before  # the folders were on the search path for the command only


def test_module_that_cannot_be_found_says_where_it_was_looked_for(project, tmp_path, capsys):
    assert _main(["validate", str(project / "model.json"), "--plugins", "hypoxia_rule"]) == 2
    err = capsys.readouterr().err
    assert "hypoxia_rule" in err and str(project) in err and "PYTHONPATH" in err
    assert err.startswith("error: ") and "Traceback" not in err  # lgca's message, not importlib's frames


def test_broken_model_file_is_reported_in_one_line(tmp_path, capsys):
    broken = tmp_path / "broken.json"
    broken.write_text('{"model": ', encoding="utf-8")
    assert _main(["validate", str(broken)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err


def test_model_without_its_plugins_says_to_import_the_module(project, capsys):
    _write_rules(project, RULES)
    assert _main(["validate", str(project / "model.json")]) == 2
    assert "--plugins hypoxia_rules" in capsys.readouterr().err


def test_bug_in_a_plugin_rule_shows_the_traceback_of_the_rule(project, tmp_path, capsys):
    _write_rules(project, BROKEN)
    assert _main(["run", str(project / "model.json"), "--plugins", "hypoxia_rules",
                  "--output", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err and "KeyError: 'rd'" in err
    assert "hypoxia_rules.py" in err and 'state.remove_cells(rates["rd"])' in err  # her file and line
    assert "step 1 was rolled back" in err


def test_error_found_by_lgca_is_one_line_with_its_notes(project, tmp_path, capsys):
    """A probability above 1 is the user's mistake, but lgca finds it: its message says what to do."""
    _write_rules(project, RULES.replace("r_d=0.2", "r_d=3.0"))
    assert _main(["run", str(project / "model.json"), "--plugins", "hypoxia_rules",
                  "--output", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err
    assert "step 1 was rolled back" in err


def test_archive_check_that_cannot_read_metadata_warns(project, tmp_path):
    _write_rules(project, RULES)
    run = tmp_path / "run"
    assert _main(["run", str(project / "model.json"), "--plugins", "hypoxia_rules", "--output", str(run)]) == 0
    (run / "metadata.json").write_text('{"provenance": {"archive": ', encoding="utf-8")  # a truncated write
    with pytest.warns(UserWarning, match="metadata.json cannot be read"):
        assert _main(["validate", str(run / "model.resolved.json")]) == 0


def test_examples_without_a_subcommand_lists_them(capsys):
    assert _main(["examples"]) == 0
    assert "random_walk" in capsys.readouterr().out


def test_help_says_what_the_positional_arguments_are(capsys):
    with pytest.raises(SystemExit):
        _main(["examples", "export", "--help"])
    out = capsys.readouterr().out
    assert "biolgca examples list" in out and ".json" in out


def test_version_option_names_the_installed_version(capsys):
    from importlib.metadata import version

    with pytest.raises(SystemExit) as exit_info:
        _main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"biolgca {version('biolgca')}"
