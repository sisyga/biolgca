import importlib
import inspect
import subprocess
import sys
from pathlib import Path

from lgca.examples import example_gallery


EXPECTED_EXAMPLES = (
    "random_walk",
    "alignment",
    "chemotaxis",
    "multispecies_birth_death",
    "identity_tumor_growth",
)


def test_each_gallery_example_has_readable_source_module():
    for name in EXPECTED_EXAMPLES:
        module = importlib.import_module(f"lgca.examples.{name}")
        source = inspect.getsource(module)

        assert hasattr(module, "INFO"), name
        assert hasattr(module, "build_spec"), name
        assert hasattr(module, "run"), name
        assert "ModelSpec(" in source, name
        assert "run_model(" in source, name
        assert module.build_spec().description.title, name


def test_gallery_cards_point_to_their_teaching_files():
    cards = {card.name: card for card in example_gallery()}

    assert tuple(cards) == EXPECTED_EXAMPLES
    for name, card in cards.items():
        assert card.source_path == f"lgca/examples/{name}.py"
        assert Path(card.source_path).exists()


def test_example_info_keeps_previous_constructor_shape():
    from lgca.examples import ExampleInfo

    info = ExampleInfo(
        name="demo",
        title="Demo",
        category="teaching",
        question="What changes?",
        concepts=("model setup",),
    )

    assert info.source_path == ""


def test_each_teaching_file_runs_a_tiny_simulation():
    for name in EXPECTED_EXAMPLES:
        module = importlib.import_module(f"lgca.examples.{name}")

        result = module.run(steps=1, showprogress=False)

        assert result.metadata["title"] == module.INFO.title
        assert result.metadata["steps"] == 1
        assert result.lgca.total_population() >= 0


def test_example_module_can_run_as_script_without_import_warning():
    completed = subprocess.run(
        [sys.executable, "-m", "lgca.examples.random_walk"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "RuntimeWarning" not in completed.stderr
    assert "Random walk example" in completed.stdout
