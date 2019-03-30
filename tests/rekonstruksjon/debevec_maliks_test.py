import numpy as np
from hdrie.rekonstruksjon import debevec_maliks

# Bildestørrelsen på orginalbildet.
IMG_SIZE = 2

# Antall eksponeringer.
N_EXPS = 4

# Skalar som brukes til å skalere eksponeringene.
EXP_SCALE = 15

# Generer et tilfeldig bilde med verdier mellom 30 og 230
ORIG_IMG = np.random.randint(200, size=(IMG_SIZE, IMG_SIZE)) + 30

# Generer ulike eksponeringsverdier som skal brukes for å lage eksponeringer av
# det orginale bildet.
EXP_OFFSETS = np.arange(-N_EXPS // 2, N_EXPS // 2) * EXP_SCALE

# Gjør om orginalbildet til en array med eksponerte bilder.
EXP_IMGS = np.array([(ORIG_IMG + offset).clip(0, 255) for offset in EXP_OFFSETS])

# Istedenfor å ha en liste med eksponerte bilder, så endrer man til en liste
# med bilder der eksponeringsverdiene er gitt for hver piksel (formatet
# funksjonen forventer).
Z = EXP_IMGS.reshape(EXP_IMGS.shape[0], -1).transpose()
DTS = np.linspace(0.01, 1, N_EXPS)


def w(z):
    return z if z <= 128 else 255 - z


def test_debevec_maliks():
    g, _ = debevec_maliks(Z, DTS, 0.1, lambda z: w(z + 1), 256)

    # Enkeltderiverte til `g`.
    dts = g[1:] - g[:-1]

    # Dobbeltderiverte til `g`.
    dtts = dts[1:] - dts[:-1]

    # Siden `g` skal være lineær, så skal alle de dobbeltderiverte være null.
    assert np.allclose(dtts, np.zeros_like(dtts))
