import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from lgca.examples import example_gallery


EXPECTED_EXAMPLES = (
    "random_walk",
    "alignment",
    "aggregation",
    "nematic_interaction",
    "chemotaxis",
    "persistent_movement",
    "contact_guidance",
    "go_and_grow",
    "go_or_grow",
    "identity_go_and_grow",
    "excitable_medium",
    "custom_rest_or_align",
    "evolutionary_go_and_grow",
    "evolutionary_go_or_grow",
    "multispecies_birth_death",
    "identity_tumor_growth",
)

NOTEBOOK_DEFAULTS = {
    "random_walk": ((50, 50), 100),
    "alignment": ((50, 50), 100),
    "aggregation": ((50, 50), 100),
    "nematic_interaction": ((50, 50), 100),
    "chemotaxis": ((50, 50), 100),
    "persistent_movement": ((12, 12), 50),
    "contact_guidance": ((50, 50), 50),
    "go_and_grow": ((50, 50), 100),
    "go_or_grow": ((50, 50), 15),
    "identity_go_and_grow": ((100,), 200),
    "excitable_medium": ((50, 50), 100),
    "custom_rest_or_align": ((50, 50), 100),
    "evolutionary_go_and_grow": ((100,), 200),
    "evolutionary_go_or_grow": ((25,), 100),
    "multispecies_birth_death": ((50, 50), 100),
    "identity_tumor_growth": ((50, 50), 50),
}


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
        assert "BioLGCA.ipynb" in module.INFO.source or "Evolutionary LGCA.ipynb" in module.INFO.source


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


@pytest.mark.parametrize("name", EXPECTED_EXAMPLES)
def test_example_defaults_match_notebook_scale(name):
    module = importlib.import_module(f"lgca.examples.{name}")
    spec = module.build_spec()

    expected_dims, expected_steps = NOTEBOOK_DEFAULTS[name]

    assert tuple(spec.space.dims) == expected_dims
    assert spec.time.steps == expected_steps


def test_each_teaching_file_runs_a_tiny_simulation():
    for name in EXPECTED_EXAMPLES:
        module = importlib.import_module(f"lgca.examples.{name}")

        result = module.run(steps=1, showprogress=False)

        assert result.metadata["title"] == module.INFO.title
        assert result.metadata["steps"] == 1
        assert result.lgca.total_population() >= 0


@pytest.mark.parametrize("name", EXPECTED_EXAMPLES)
def test_example_module_can_run_from_examples_folder(name):
    examples_dir = Path(__file__).parents[1] / "lgca" / "examples"
    completed = subprocess.run(
        [sys.executable, f"{name}.py", "--steps", "1"],
        check=False,
        capture_output=True,
        text=True,
        cwd=examples_dir,
    )

    assert completed.returncode == 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert importlib.import_module(f"lgca.examples.{name}").INFO.title in completed.stdout


def test_example_module_can_run_with_python_m():
    completed = subprocess.run(
        [sys.executable, "-m", "lgca.examples.random_walk", "--steps", "1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "RuntimeWarning" not in completed.stderr
    assert "Random walk example" in completed.stdout
