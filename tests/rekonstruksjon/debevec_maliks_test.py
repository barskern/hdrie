import numpy as np
import hdrie.rekonstruksjon as rekons

import pytest
from pytest import approx


def scale(m, r=(0, 1)):
    new_min, new_max = r
    return new_min + ((m - m.min()) * new_max) / (m.max() - m.min())


def test_responskurve_feil_dim():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [
            (orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8)
            for dt in eksp_tider[1:]  # Fjerner en delta tid
        ]
    )

    with pytest.raises(ValueError):
        rekons.responskurve(eksp_bilder, eksp_tider, 100, 256)


def test_responskurve_retur():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    lg, lE = rekons.responskurve(eksp_bilder, eksp_tider, 100, 256)

    assert lg.shape == (256,)
    assert lE.shape == (5, 5)


def test_responskurve_feil_antall_verdier():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    with pytest.raises(IndexError) as e:
        rekons.responskurve(eksp_bilder, eksp_tider, 100, 100)

    assert "size 100" in str(e.value)


def test_responskurve_vekter():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    vekter = np.exp(
        np.concatenate((np.linspace(0, 4, 256 // 2), np.linspace(4, 0, 256 // 2)))
    )

    # Kall funksjonen for å generere
    lg, _ = rekons.responskurve(eksp_bilder, eksp_tider, 10, 256, vekter)
    g = np.exp(lg)

    dgdt = g[1:] - g[:-1]  # Enkeltderiverte til `g`.
    dgdtt = dgdt[1:] - dgdt[:-1]  # Dobbeltderiverte til `g`.

    # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
    # Pga. at det blir noen unøyaktigheter rundt endene på `g`, så benytter vi
    # kun sentrerte verdier i sammenlikningen.
    senter = dgdtt[10:-30]

    assert pytest.approx(0.0, abs=1e-3) == senter


def test_responskurve_enkel():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    # Kall funksjonen for å generere
    lg, _ = rekons.responskurve(eksp_bilder, eksp_tider, 100, 256)
    g = np.exp(lg)

    dgdt = g[1:] - g[:-1]  # Enkeltderiverte til `g`.
    dgdtt = dgdt[1:] - dgdt[:-1]  # Dobbeltderiverte til `g`.

    # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
    # Pga. at det blir noen unøyaktigheter rundt endene på `g`, så benytter vi
    # kun sentrerte verdier i sammenlikningen.
    senter = dgdtt[10:-30]
    assert approx(0.0, abs=1e-3) == senter


def test_rekonstruer_irradiens():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    # Regn ut responskurven
    lg, _ = rekons.responskurve(eksp_bilder, eksp_tider, 100, 256)
    lE = rekons.irradians(eksp_bilder, eksp_tider, lg, 256)
    kons = scale(np.exp(lE))

    # Både median og varians av forskjellen bør være svært små.
    assert approx(0.0, abs=1e-1) == np.abs(kons - orig_img).mean()
    assert approx(0.0, abs=1e-2) == np.abs(kons - orig_img).var()
