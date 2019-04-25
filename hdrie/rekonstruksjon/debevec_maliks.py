"""
Definerer Debevec-Maliks metode for å rekonstruere responsverdiene til et sett
med bilder for å kunne gjennskape et bilde med høy dynamisk radians.
"""

import numpy as np


def debevec_maliks(eksp_bilder, eksp_tider, glatthet, antall_verdier, vekter=None):
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
    antall_verdier : integer
        Antall ulike verdier (verdier går fra 0 til antall_verdier - 1).
    vekter : {(antall_verdier,)} ndarray, optional
        Vektfunksjon som returner et vekttall basert på pikselverdi. Dersom den
        ikke er spesifisert så blir den definert som absoluttverdien av
        avstanden til midten av pikselverdiene.

    Returnerer
    ----------
    g : (antall_verdier,) ndarray
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

    antall_piksler = eksp_bilder.shape[1]

    if vekter is None:
        vekter = np.concatenate(
            (
                np.arange(1, 1 + antall_verdier // 2),
                np.arange(antall_verdier // 2, 0, -1),
            )
        )

    # Logaritmen av eksponseringstidene (slik at brukeren slipper).
    log_eksp_tider = np.log(eksp_tider)

    # start snippet debevec-maliks-algo

    A = np.zeros(
        (
            antall_eksp * antall_piksler + antall_verdier - 1,
            antall_verdier + antall_piksler,
        )
    )
    b = np.zeros(A.shape[0])

    # Indeksen til irradiansene i likningssettet.
    irradians_indeks = np.arange(antall_piksler) + antall_verdier

    # Indeksen inn i matrisen for likningene.
    k = np.arange(antall_piksler)

    # For hver eksponering av bildet og tilhørende logaritmisk eksponeringstid.
    for ln_eksp, piksler in zip(log_eksp_tider, eksp_bilder):
        # Alle responskurvene for disse plasseringene har `w(z)`.
        A[k, piksler] = vekter[piksler]

        # Alle irradiansene for disse plasseringene har `-w(z)`.
        A[k, irradians_indeks] = -vekter[piksler]

        # Alle tidene for disse plasseringene har `w(z) * ln dt`.
        b[k] = vekter[piksler] * ln_eksp

        # Inkrementer til neste del av matrisen for neste eksponering.
        k += antall_piksler

    # Fiks kurven ved å sette midtpunktet til 0.
    A[antall_eksp * antall_piksler, antall_verdier // 2] = 1

    # Lag indeksering for alle verdier (eksl. ytterpunkter) i 3 dimensjoner.
    i = np.arange(1, antall_verdier - 1)[:, np.newaxis].repeat(3, axis=1)

    # Endre `k` til å indeksere over alle verdier (utenom ytterpunkter) med
    # offsett fra tidligere definert likning.
    k = i + (antall_piksler * antall_eksp)

    # Sørg for at kurven er glatt ved å legge til ledd som prøver å minimere
    # g''(z), altså få kurven mest mulig glatt.
    A[k, i + [-1, 0, 1]] = vekter[i] * glatthet * [1, -2, 1]

    # Løs likningssettet.
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    return x[:antall_verdier], x[antall_verdier:].reshape(orginal_verdi_form)

    # end snippet debevec-maliks-algo


def rekonstruer_irradians(
    eksp_bilder, eksp_tider, responskurve, antall_verdier, vekter=None
):
    """
    Gitt et sett med pikselverdier observert over flere bilder med ulik
    eksponering og responskurven til bildene, returner den rekonstruerte
    irradiansen ved hver enkelt pikselposisjon.

    Parametere
    ----------
    eksp_bilder : {(E, I), (E, X, Y)} ndarray
        Ulike eksponeringer (j) av pikselverdier (i) eller (x, y).
    eksp_tider : (E,) ndarray
        Eksponeringstid av bildet med eksponering (j).
    responskurve : (antall_verdier,) ndarray
        Responskurven til bildene.
    antall_verdier : integer
        Antall ulike pikselverdier (antas at pikselverdier går fra 0 til
        antall_verdier - 1).
    vekter : {(antall_verdier,)} ndarray, optional
        Vektfunksjon som returner et vekttall basert på pikselverdi. Dersom den
        ikke er spesifisert så blir den definert som absoluttverdien av
        avstanden til midten av pikselverdiene.

    Returnerer
    ----------
    lE : {(I,), (X, Y)} ndarray
        Logaritmen til irradiansen til pikslen ved gitt posisjon.
    """
    if vekter is None:
        vekter = np.concatenate(
            (
                np.arange(1, 1 + antall_verdier // 2),
                np.arange(antall_verdier // 2, 0, -1),
            )
        )

    if len(eksp_bilder.shape) == 2:
        # Istedenfor (E, I), så gjør vi om til (I, E).
        eksp_bilder = eksp_bilder.transpose(1, 0)
    elif len(eksp_bilder.shape) == 3:
        # Istedenfor (E, X, Y), så gjør vi om til (X, Y, E).
        eksp_bilder = eksp_bilder.transpose(1, 2, 0)
    else:
        raise ValueError("eksp_bilder må ha enten 2 eller 3 dimensjoner")

    return (vekter[eksp_bilder] * (responskurve[eksp_bilder] - np.log(eksp_tider))).sum(
        -1
    ) / vekter[eksp_bilder].sum(-1)
