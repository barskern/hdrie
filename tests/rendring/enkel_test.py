import numpy as np
import hdrie.rendring as rend
import pytest
from pytest import approx


def test_gamma_global():
    orig_bilde = np.random.rand(10, 10, 3) ** 3
    kons = rend.gamma(orig_bilde, 0.5)

    assert np.allclose(kons, orig_bilde ** 0.5)


def test_gamma_global_feil():
    orig_bilde = np.random.rand(10, 10, 3) ** 3

    with pytest.raises(ValueError) as e:
        rend.gamma(orig_bilde, 2)

    assert "mellom 0 og 1" in str(e.value)


def test_gamma_luminans():
    orig_bilde = np.ones((10, 10, 3))
    gbilde = rend.gamma_luminans(orig_bilde)

    assert approx((3 ** 0.5) * (1 / 3)) == gbilde


def test_gamma_kombo_lumi():
    orig_bilde = np.ones((10, 10, 3))
    gbilde = rend.gamma_kombo(orig_bilde, 1)
    expt = rend.gamma_luminans(orig_bilde)

    assert np.allclose(expt, gbilde)


def test_gamma_kombo_glob():
    orig_bilde = np.ones((10, 10, 3))
    gbilde = rend.gamma_kombo(orig_bilde, 0)
    expt = rend.gamma(orig_bilde)

    assert np.allclose(expt, gbilde)


def test_gamma_kombo():
    orig_bilde = np.ones((10, 10, 3))
    gbilde = rend.gamma_kombo(orig_bilde, 0.5)
    expt = 0.5 * rend.gamma_luminans(orig_bilde) + 0.5 * rend.gamma(orig_bilde)

    assert np.allclose(expt, gbilde)


def test_gamma_kombo_fail():
    orig_bilde = np.ones((10, 10, 3))
    with pytest.raises(ValueError) as e:
        gbilde = rend.gamma_kombo(orig_bilde, 2)

    assert "mellom 0 og 1" in str(e.value)
