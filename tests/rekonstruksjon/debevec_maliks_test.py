import numpy as np
from hdrie.rekonstruksjon import debevec_maliks, debevec_maliks_color


def w(z):
    return z if z <= 128 else 255 - z


def test_debevec_maliks():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    mini, maks = (50, 200)
    orig_img = np.random.randint(maks - mini, size=(5, 5), dtype=np.uint8) + mini
    exps = np.arange(-3, 3) * 10
    exp_imgs = np.array([(orig_img + offset) for offset in exps])
    dts = np.linspace(0.01, 1, 6)

    # Istedenfor å ha en liste med eksponerte bilder, så endrer man til en liste
    # med bilder der eksponeringsverdiene er gitt for hver piksel (formatet
    # funksjonen forventer).
    Z = exp_imgs.reshape(exp_imgs.shape[0], -1).transpose()

    g, _ = debevec_maliks(Z, dts, 0.1, w, 256)

    # Enkeltderiverte til `g`.
    dgdt = g[1:] - g[:-1]

    # Dobbeltderiverte til `g`.
    dgdtt = dgdt[1:] - dgdt[:-1]

    # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
    assert np.allclose(dgdtt, np.zeros_like(dgdtt))


def test_debevec_maliks_color():
    # Lag testdata (tilfeldig bilde + lineære eksponeringer).
    mini, maks = (50, 200)
    orig_img = np.random.randint(maks - mini, size=(5, 5, 3), dtype=np.uint8) + mini
    exps = np.arange(-3, 3) * 10
    exp_imgs = np.array([(orig_img + offset) for offset in exps])
    dts = np.linspace(0.01, 1, 6)

    # Istedenfor å ha en liste med eksponerte bilder, så endrer man til en liste
    # med bilder der eksponeringsverdiene er gitt for hver piksel (formatet
    # funksjonen forventer).
    Z = exp_imgs.reshape(exp_imgs.shape[0], exp_imgs.shape[1] ** 2, -1).transpose()

    # Sjekk at hver eneste kanal har en lineær responskurve.
    for g, _ in debevec_maliks_color(Z, dts, 0.1, w, 256):
        # Enkeltderiverte til `g`.
        dgdt = g[1:] - g[:-1]

        # Dobbeltderiverte til `g`.
        dgdtt = dgdt[1:] - dgdt[:-1]

        # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
        assert np.allclose(dgdtt, np.zeros_like(dgdtt))
