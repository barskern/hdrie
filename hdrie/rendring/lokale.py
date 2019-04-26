from scipy.ndimage.filters import gaussian_filter
from .globale import gamma
import numpy as np


def lin_spat_filter(bilde, render=gamma, sigma=2):
    """
    Gitt et HDR-bilde og en rendringsfunksjon, returnerer resultatet av å
    gjennomføre en lineær spatiell filtrering av bildet og anvende
    rendringsfunksjonen på de lave frekvensene.

    Parameters
    ----------
    bilde : ndarray
        Et bilde som kan ha ubegrenset antall dimensjoner/kanaler.
    render : function, optional
        Funksjon som tar et bilde og returner en HDR-rendring av bildet.
    sigma : <0, inf> float, optional
        Styrken til lavpassfilteret.

    Returns
    -------
    gbilde : ndarray
        Bildet etter å ha anvendt rendringsfunksjonen på de lave frekvensene.
    """
    lav_pass = gaussian_filter(bilde, sigma=(sigma, sigma, 0))
    return bilde - lav_pass + render(lav_pass)


# start snippet ikke-lin-spat-filter


def ikke_lin_spat_filter(
    bilde, render=gamma, sigma_avstand=0.1, sigma_intensitet=0.1, storrelse=5
):
    """
    Gitt et HDR-bilde og en rendringsfunksjon, returnerer resultatet av å
    gjennomføre en ikke lineær spatiell filtrering av bildet og anvende
    rendringsfunksjonen på de lave frekvensene.

    Parameters
    ----------
    bilde : ndarray
        Et bilde som kan ha ubegrenset antall dimensjoner/kanaler.
    render : function, optional
        Funksjon som tar et bilde og returner en HDR-rendring av bildet.
    sigma_avstand : <0, inf> float, optional
        Sigma for avstandene (relative avstander).
    sigma_intensitet : <0, inf> float, optional
        Sigma for intensiteten (verdiene).
    storrelse : int <3, inf> integer
        Størrelsen på kjernen.

    Returns
    -------
    gbilde : ndarray
        Bildet etter å ha anvendt rendringsfunksjonen på de lave frekvensene.
    """
    halv_s = storrelse // 2

    def gauss_func(sigma):
        def func(e, x=0):
            d = np.abs(e - x) / sigma
            return np.exp(-0.5 * d * d)

        return func

    linje = np.arange(0, storrelse) - halv_s

    # Regn ut relative avstander fra sentrum av et matrise.
    avstander_mono = np.sqrt(
        ((np.mgrid[:storrelse, :storrelse] - halv_s) ** 2).sum(axis=0)
    )
    # Gjør om avstandene til å være basert på en gauss-funksjon.
    gauss_avstander = gauss_func(sigma_avstand)(
        np.repeat([avstander_mono], 3, axis=0).transpose(1, 2, 0)
    )

    # Lag avanserte indekser slik at man slipper å kjøre løkker i python. Dette
    # gjør koden betydelig raskere.
    rad, col = np.mgrid[:storrelse, :storrelse] - halv_s
    brad, bcol = np.mgrid[: bilde.shape[0], : bilde.shape[1]]
    y_indeks = np.add.outer(brad, rad).clip(0, bilde.shape[0] - 1)
    x_indeks = np.add.outer(bcol, col).clip(0, bilde.shape[1] - 1)

    # Regn ut absolutte intensitetsforskjellen for hver enkelt kjerne og send
    # resultatet gjennom en gauss-funksjon av den Euklidiske avstanden.
    gauss_intensiteter = gauss_func(sigma_intensitet)(
        bilde[y_indeks, x_indeks], bilde[..., np.newaxis, np.newaxis, :]
    )

    # Kombiner intensitetene og avstandene.
    kombinert = gauss_intensiteter * gauss_avstander

    # Kjør bildet gjennom det bilaterale filteret.
    m = (bilde[y_indeks, x_indeks] * kombinert).sum(axis=(2, 3)) / kombinert.sum(
        axis=(2, 3)
    )

    # Kjør resulatet av filteret gjennom rendringsfunksjonen og legg til
    # detaljene.
    return bilde - m + render(m)


# end snippet ikke-lin-spat-filter
