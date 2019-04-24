import numpy as np
from hdrie.rekonstruksjon import debevec_maliks
from pytest import approx


def test_debevec_maliks():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    mini, maks = (50, 200)
    orig_img = np.random.rand(5, 5)
    eksp_tider = np.arange(1, 6)
    eksp_bilder = np.array(
        [(orig_img * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    # Kall funksjonen for å generere
    g, _ = debevec_maliks(eksp_bilder, eksp_tider, 30, 256)
    g = np.exp(g)

    dgdt = g[1:] - g[:-1]  # Enkeltderiverte til `g`.
    dgdtt = dgdt[1:] - dgdt[:-1]  # Dobbeltderiverte til `g`.

    # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
    # Pga. at det blir noen unøyaktigheter rundt endene på `g`, så benytter vi
    # kun sentrerte verdier i sammenlikningen.
    senter = dgdtt[50:-50]
    assert approx(0.0, abs=1e-5) == senter.sum() / len(senter)


def test_debevec_maliks_flere_kanaler():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    mini, maks = (50, 200)
    orig_img = np.random.rand(5, 5, 3)
    eksp_tider = np.arange(1, 6)
    eksp_bilder = np.array(
        [(orig_img * dt * 255).clip(0, 255).astype(np.uint8) for dt in eksp_tider]
    )

    for kanal in eksp_bilder.transpose(3, 0, 1, 2):
        g, _ = debevec_maliks(kanal, eksp_tider, 30, 256)
        g = np.exp(g)

        dgdt = g[1:] - g[:-1]  # Enkeltderiverte til `g`.
        dgdtt = dgdt[1:] - dgdt[:-1]  # Dobbeltderiverte til `g`.

        # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
        # Pga. at det blir noen unøyaktigheter rundt endene på `g`, så benytter vi
        # kun sentrerte verdier i sammenlikningen.
        senter = dgdtt[50:-50]
        assert approx(0.0, abs=1e-5) == senter.sum() / len(senter)
