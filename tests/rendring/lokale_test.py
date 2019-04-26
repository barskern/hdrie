import numpy as np
from hdrie.rendring import lin_spat_filter, ikke_lin_spat_filter
from pytest import approx


def test_lin_spat_filter():
    orig_bilde = np.random.rand(10, 10, 3) ** 3
    kons = lin_spat_filter(orig_bilde)

    diff = np.abs(orig_bilde ** 0.5 - kons)

    assert approx(0.0, abs=4e-1) == diff


def test_ikke_lin_spat_filter():
    orig_bilde = np.random.rand(10, 10, 3) ** 3
    kons = ikke_lin_spat_filter(orig_bilde)

    diff = np.abs(orig_bilde ** 0.5 - kons)

    assert approx(0.0, abs=4e-1) == diff
