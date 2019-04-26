import numpy as np


def gamma(bilde, g=0.5):
    """
    Gitt et HDR-bilde, returnerer resultatet av å gjennomføre enkel
    rendring ved hjelp av en gamma-funksjon.

    Parameters
    ----------
    bilde : ndarray
        Et bilde som kan ha ubegrenset antall dimensjoner/kanaler.
    g : <0, 1> float
        Gamma-verdien som brukes for å rendre bildet.

    Returns
    -------
    gbilde : ndarray
        Bildet etter å ha anvendt gammafunksjonen.
    """
    if 0.0 < g and g < 1.0:
        return bilde ** g
    else:
        raise ValueError("gammaverdien må være mellom 0 og 1")


def gamma_luminans(bilde, g=0.5):
    """
    Gitt et HDR-bilde, returnerer resultatet av å gjennomføre enkel
    rendring ved hjelp av en gamma-funksjon på luminansen til bildet.

    Parameters
    ----------
    bilde : {(X, Y, 3)} ndarray
        Et bilde med 3 fargekanaler.
    g : <0, 1> float
        Gamma-verdien som brukes for å rendre bildet.

    Returns
    -------
    gbilde : ndarray
        Bildet etter å ha anvendt gammafunksjonen på luminansen.
    """
    # Regn ut luminansen (R + G + B).
    L = bilde.sum(axis=2)
    return gamma(L[:, :, np.newaxis]) * (bilde / L[:, :, np.newaxis])


def gamma_kombo(bilde, v=0.5, g=0.5):
    """
    Gitt et HDR-bilde, returnerer resultatet av å gjennomføre enkel
    rendring ved hjelp av en vektet sum av gamma-funksjon på luminansen
    og gamma-funksjonen på alle kanalene i bildet.

    Parameters
    ----------
    bilde : {(X, Y, 3)} ndarray
        Et bilde med 3 fargekanaler.
    v : <0, 1> float
	Bestemmer hvor mye luminansen vektes (`v * gamma_luminans + (1 - v) *
        gamma`).
    g : <0, 1> float
        Gamma-verdien som brukes for å rendre bildet.

    Returns
    -------
    gbilde : ndarray
        Bildet etter å ha anvendt gammafunksjonen.
    """
    if 0.0 <= v and v <= 1.0:
        return v * gamma_luminans(bilde, g) + (1 - v) * gamma(bilde, g)
    else:
        raise ValueError("vektentallet må være mellom 0 og 1")
