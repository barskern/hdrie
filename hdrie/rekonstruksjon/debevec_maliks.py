"""
Definerer Debevec-Maliks metode for å rekonstruere responsverdiene til et sett
med bilder for å kunne gjennskape et bilde med høy dynamisk radians.
"""

import numpy as np

# start snippet debevec-maliks-resp


def responskurve(eksp_bilder, eksp_tider, glatthet, antall_verdier, vekter=None):
    """
    Gitt et sett med pikselverdier observert over flere bilder med ulik
    eksponering, returner den logaritmiske responskurven til bildesettet og
    logaritmen til irradiansen til de observerte pikslene.

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
        Logaritmen til responskurven til pikselverdiene.
    lE : {(I,), (X, Y)} ndarray
        Logaritmen til irradiansen til pikslen ved gitt posisjon.
    """

    antall_eksp = eksp_bilder.shape[0]
    if not len(eksp_tider) == antall_eksp:
        raise ValueError(
            "eksp_tider må ha like mange verdier som det er eksponeringer i eksp_bilder"
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
    # Lag indeksering for alle likningene i `A` som skal settes.
    k = i + (antall_piksler * antall_eksp)
    # Gjør kurven glatt ved å legge til ledd som prøver å minimere g''(z).
    A[k, i + [-1, 0, 1]] = vekter[i] * glatthet * [1, -2, 1]

    # Løs likningssettet for å finne den best tilpassede responskurven.
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    return x[:antall_verdier], x[antall_verdier:].reshape(orginal_verdi_form)


# end snippet debevec-maliks-resp

# start snippet debevec-maliks-irrad


def irradians(eksp_bilder, eksp_tider, res_kurve, antall_verdier, vekter=None):
    """
    Gitt et sett med pikselverdier observert over flere bilder med ulik
    eksponering og den logaritmiske responskurven til bildene, returner den
    rekonstruerte irradiansen ved hver enkelt pikselposisjon.

    Parametere
    ----------
    eksp_bilder : {(E, I), (E, X, Y)} ndarray
        Ulike eksponeringer (j) av pikselverdier (i) eller (x, y).
    eksp_tider : (E,) ndarray
        Eksponeringstid av bildet med eksponering (j).
    res_kurve : (antall_verdier,) ndarray
        Den logaritmiske responskurven til bildene.
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
        Logaritmen til irradiansen til bildene.
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

    s = (vekter[eksp_bilder] * (res_kurve[eksp_bilder] - np.log(eksp_tider))).sum(-1)
    d = vekter[eksp_bilder].sum(-1)

    # For å forhindre deling på 0, så endres 0 til 1.
    d[d == 0] = 1

    return s / d


# end snippet debevec-maliks-irrad
