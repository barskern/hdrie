"""
Definerer Debevec-Maliks metode for å rekonstruere responsverdiene til et sett
med bilder for å kunne gjennskape et bilde med høy dynamisk radians.
"""

import numpy as np


def debevec_maliks(Z, dT, l, w):
    """
    Gitt et sett med pikselverdier (i) observert over flere bilder med ulik
    eksponering (j), returner responskurven til bildesettet og logaritmen til
    irradiansen til de observerte pikslene.

    Parametere
    ----------

    Z : 2-dim array
        Pikselverdiene (i) som er observert med ulik eksponering (j).
    dT : array
        Logaritmisk delta tid for ekspoering (j).
    l : float
        Konstant som bestemmer mengden glatthet.
    w : function
        Returnerer et vekttall basert på gitt z.

    Returnerer
    ----------

    (g, lE) : (array, array)
        Returnerer en tuple der første er responskurven til bildesettet og den
        andre delen er den logaritmiske irradiansen til pikslene.
    """
    n = 255

    A = np.zeros((Z.shape[0] * Z.shape[1] + n, n + Z.shape[0]))
    b = np.zeros(A.shape[0])

    # Sett inn de data-tilpassende funksjonene
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w(Z[i, j])

            A[k, Z[i, j]] = wij
            A[k, n+i] = -wij

            b[k] = wij * dT[j]

            k += 1

    # Fiks kurven ved å sette midtpunktet til 0
    A[k, 128] = 0
    k += 1

    # Ta hensyn til glattheten
    for i in range(n-2):
        A[k, i] = l * w(i)
        A[k, i + 1] = -2 * l * w(i)
        A[k, i + 2] = l * w(i)

        k += 1

    # Løs likningssettet
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    return x[:n], x[n:]
