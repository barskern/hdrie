"""
Definerer Debevec-Maliks metode for å rekonstruere responsverdiene til et sett
med bilder for å kunne gjennskape et bilde med høy dynamisk radians.
"""

import numpy as np


def debevec_maliks(
    eksp_bilder, eksp_tider, glatthet, antall_pikselverdier, vekter=None
):
    """
    Gitt et sett med pikselverdier observert over flere bilder med ulik
    eksponering, returner responskurven til bildesettet og logaritmen til
    irradiansen til de observerte pikslene.

    Parametere
    ----------
    eksp_bilder : {(E, I), (E, X, Y)} ndarray
        Ulike eksponeringer (j) av pikselverdier (i) eller (x, y).
    eksp_tider : (E,) ndarray
        Eksponeringstid av bildet med eksponering (j).
    glatthet : float
        Konstant som bestemmer mengden glatthet.
    antall_pikselverdier : integer
        Antall ulike pikselverdier (antar at pikselverdier går fra 0 til
        antall_pikselverdier - 1).
    vekter : {(antall_pikselverdier,)} ndarray, optional
        Vektfunksjon som returner et vekttall basert på pikselverdi. Dersom den
        ikke er spesifisert så blir den definert som absoluttverdien av
        avstanden til midten av pikselverdiene.

    Returnerer
    ----------
    g : (antall_pikselverdier,) ndarray
        Responskurven til pikselverdiene.
    lE : {(I,), (X, Y)} ndarray
        Logaritmen til irradiansen til pikslen ved gitt posisjon.
    """

    antall_eksp = eksp_bilder.shape[0]
    if not len(eksp_tider) == antall_eksp:
        raise ValueError(
            "eksp_tider må ha nøyaktig like mange verdier som det er eksponeringer i eksp_bilder"
        )
    # Tar vare på den gamle formen til plasseringen slik at man kan endre de
    # logaritmiske irradiansene til å ha samme form som inngangsverdien.
    orginal_verdi_form = eksp_bilder.shape[1:]

    if len(eksp_bilder.shape) == 3:
        # Gjør om fra (eksponering, x, y) til (eksponering, plassering).
        eksp_bilder = eksp_bilder.reshape(antall_eksp, -1)
    elif len(eksp_bilder.shape) < 2 or 3 < len(eksp_bilder.shape):
        raise ValueError("eksp_bilder må ha enten 2 eller 3 dimensjoner")

    antall_verdier = eksp_bilder.shape[1]

    if vekter is None:
        vekter = np.concatenate(
            (
                np.arange(0, antall_pikselverdier // 2),
                np.arange(antall_pikselverdier // 2, 0, -1),
            )
        )

    # Logaritmen av eksponseringstidene (slik at brukeren slipper).
    log_eksp_tider = np.log(eksp_tider)

    # start snippet debevec-maliks-algo

    A = np.zeros(
        (
            antall_eksp * antall_verdier + antall_pikselverdier - 1,
            antall_pikselverdier + antall_verdier,
        )
    )
    b = np.zeros(A.shape[0])

    # Sett inn de data-tilpassende funksjonene.
    k = 0
    # For hver eksponering av bildet og tilhørende eksponeringstid.
    for eksp, piksler in zip(log_eksp_tider, eksp_bilder):
        # For hver pikselverdi og tilhørende indeks.
        for i, verdi in enumerate(piksler):
            A[k, verdi] = vekter[verdi]
            A[k, antall_pikselverdier + i] = -vekter[verdi]

            b[k] = vekter[verdi] * eksp

            k += 1

    # Fiks kurven ved å sette midtpunktet til 0.
    A[k, antall_pikselverdier // 2] = 1
    k += 1

    # Sørg for at kurven er glatt ved å legge til ledd som prøver å minimere
    # g''(z), altså få kurven mest mulig glatt.
    for i in range(1, antall_pikselverdier - 1):
        A[k, i] = -2 * glatthet * vekter[i]
        A[k, i - 1] = A[k, i + 1] = glatthet * vekter[i]

        k += 1

    # Løs likningssettet.
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    return (
        x[:antall_pikselverdier],
        x[antall_pikselverdier:].reshape(orginal_verdi_form),
    )

    # end snippet debevec-maliks-algo
