"""Self-checks of the frozen reference kernels in ``tests/legacy/identity_kernels.py``.

These test the oracle, not the package: the legacy identity-based rules in
``tests/legacy`` use these kernels, and comparison tests check biolgca's rules
against those rules. The package's own truncated-normal mutation and trait
inheritance are tested in ``birth_death_test.py``.
"""

import numpy as np

from lgca import get_lgca


def test_reference_birth_kernel_preserves_ids_and_indexes_new_properties():
    from tests.legacy.identity_kernels import apply_identity_birth

    lgca = get_lgca(geometry="lin", ib=True, nodes=np.array([[1, 0], [2, 3], [0, 0]]),
                    interaction="only_propagation", seed=125)
    lgca.props["r_b"] = [0., 1., 1., 1.]
    for _ in range(10):
        apply_identity_birth(lgca, a_max=1, std=.01)
        lgca.update_dynamic_fields()
    labels = lgca.nodes[lgca.nonborder]
    living = labels[labels > 0]
    assert set(living) == {1, 2, 3, 4}
    assert len(living) == len(set(living))
    assert not labels[2].any()
    assert len(lgca.props["r_b"]) == int(lgca.maxlabel) + 1 == 5
    assert 0 <= lgca.props["r_b"][4] <= 1


def test_reference_truncated_normal_sampler_respects_bounds_and_moments():
    from tests.legacy.identity_kernels import sample_truncated_normal

    rng = np.random.default_rng(3)
    means = np.repeat([0.0, 0.5, 1.5], 20000)
    samples = sample_truncated_normal(0.0, 1.0, means, 0.05, rng)

    assert samples.shape == means.shape
    assert np.all((samples >= 0) & (samples <= 1))
    centred = samples[means == 0.5]
    np.testing.assert_allclose(centred.mean(), 0.5, atol=0.002)
    np.testing.assert_allclose(centred.std(), 0.05, rtol=0.03)
    # A normal truncated at its mean is half-normal with mean sigma * sqrt(2 / pi).
    np.testing.assert_allclose(samples[means == 0.0].mean(), 0.05 * np.sqrt(2 / np.pi), rtol=0.03)
    assert samples[means == 1.5].min() > 0.9
    np.testing.assert_array_equal(sample_truncated_normal(0.0, 1.0, [0.2, 2.0], 0.0, rng), [0.2, 2.0])


def test_reference_daughter_properties_copy_untouched_traits_from_parents():
    from types import SimpleNamespace

    from tests.legacy.identity_kernels import append_daughter_properties

    lgca = SimpleNamespace(maxlabel=5, props={
        "r_b": [0.0, 0.1, 0.2, 0.3],
        "family": [0, 1, 1, 2],
        "kappa": np.array([0.0, 4.0, 5.0, 6.0]),
    })

    append_daughter_properties(lgca, [3, 1], r_b=[0.35, 0.15])

    assert lgca.props["r_b"] == [0.0, 0.1, 0.2, 0.3, 0.35, 0.15]
    assert lgca.props["family"] == [0, 1, 1, 2, 2, 1]
    np.testing.assert_array_equal(lgca.props["kappa"], [0, 4, 5, 6, 6, 4])
