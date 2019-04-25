import numpy as np
from hdrie.rekonstruksjon import debevec_maliks, rekonstruer_irradians
from pytest import approx


def scale(m, r=(0, 1)):
    new_min, new_max = r
    return new_min + ((m - m.min()) * new_max) / (m.max() - m.min())


def test_debevec_maliks():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    # Kall funksjonen for å generere
    g, _ = debevec_maliks(eksp_bilder, eksp_tider, 100, 256)
    g = np.exp(g)

    dgdt = g[1:] - g[:-1]  # Enkeltderiverte til `g`.
    dgdtt = dgdt[1:] - dgdt[:-1]  # Dobbeltderiverte til `g`.

    # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
    # Pga. at det blir noen unøyaktigheter rundt endene på `g`, så benytter vi
    # kun sentrerte verdier i sammenlikningen.
    senter = dgdtt[20:-30]
    assert approx(0.0, abs=1e-3) == senter


def test_debevec_maliks_flere_kanaler():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5, 3) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    for kanal in eksp_bilder.transpose(3, 0, 1, 2):
        g, _ = debevec_maliks(kanal, eksp_tider, 100, 256)
        g = np.exp(g)

        dgdt = g[1:] - g[:-1]  # Enkeltderiverte til `g`.
        dgdtt = dgdt[1:] - dgdt[:-1]  # Dobbeltderiverte til `g`.

        # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
        # Pga. at det blir noen unøyaktigheter rundt endene på `g`, så benytter vi
        # kun sentrerte verdier i sammenlikningen.
        senter = dgdtt[20:-30]
        assert approx(0.0, abs=1e-3) == senter


def test_rekonstruer_irradiens():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    orig_img = np.random.rand(5, 5) ** 3
    eksp_tider = 2 ** np.arange(1, 5)
    eksp_bilder = np.array(
        [(orig_img * 0.1 * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    # Regn ut responskurven
    g, _ = debevec_maliks(eksp_bilder, eksp_tider, 100, 256)
    lE = rekonstruer_irradians(eksp_bilder, eksp_tider, g, 256)
    kons = scale(np.exp(lE))

    # Både median og varians av forskjellen bør være svært små.
    assert approx(0.0, abs=1e-1) == np.abs(kons - orig_img).mean()
    assert approx(0.0, abs=1e-2) == np.abs(kons - orig_img).var()
