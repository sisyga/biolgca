"""Parameter combinations for tests that vary three settings at once."""


def pairwise(first, second, third):
    """Every pair of values of any two settings, instead of every combination of all three.

    ``first`` and ``second`` are crossed in full; ``third`` cycles so that each of its values meets every
    value of ``first`` and every value of ``second``. For tests whose third setting runs code that does
    not depend on the other two together (checked with per-test branch coverage when introduced).
    """
    if len(third) > min(len(first), len(second)):
        raise ValueError("the third setting may have no more values than each of the other two")
    return [(a, b, third[(i + j) % len(third)]) for i, a in enumerate(first) for j, b in enumerate(second)]
