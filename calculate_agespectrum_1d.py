# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:07:00 2020

@author: anengel

last modified by Thomas Wagenhäuser 2022-09-22
"""
# Standard library imports
import numpy as np
import matplotlib.pyplot as plt


def Calculate_AgeSpectrum_1D(m_age, rom, nyy=30, nmnth=12, per_month=10, plot=False):
    """Calculate transit time probability distribution.

    based on Calculate_AgeSpectrum_1D by Harald Boenisch written in IDL
    Age spectrum for 1-D flow advection model with diffusion :

    G(t) = m_age³/(4*pi*w_age²*t³) * exp[ -m_age*(t-m_age)²/(4*w_age²*t) ]

    Parameters:
    --------
    m_age : float
        mean age is the first moment of the age spectrum distribution
    w_age : float
        width age is the second moment of the m_age centered age
        spectrum distribution
    rom : float
        ratio of moments
    nyy : int, optional
        Number of years over which to calculate the age spectrum. The default is 30.
    nmth : int, optional
        Number of months to use (should alway be 12). The default is 12.
    per_month: int, optional
        Number of data Points per Month. The default is 10.
    plot : bool, optional
        set to true if you want a plot of the spectrum. The default is False.

    Returns:
    --------
    Agespectrum G, 2 dimensional np.array. G[:, 0] contains the transit times,
    G[:, 1] contains the probabilities

    """
    w = (rom * m_age) ** 0.5

    nt = nyy * nmnth * per_month
    dt = 1 / (nt / nyy)
    t = dt * (np.arange(nt) + 1)

    # store transit time axis and probabilities together in one array
    G = np.zeros((len(t) + 1, 2))
    G[1:, 0] = t

    # calculate probabilities
    G[1:, 1] = (m_age ** 3 / (4 * np.pi * w ** 2 * t ** 3)) ** 0.5 * np.exp(
        -m_age * (t - m_age) ** 2 / (4 * w ** 2 * t)
    )

    # normalize age spectrum
    G[:, 1] = G[:, 1] / np.trapz(G[:, 1], x=G[:, 0])

    if plot:
        fig, ax = plt.subplots()
        ax.plot(G[:, 0], G[:, 1])
        ax.set(
            xlabel="transit time (years)",
            ylabel="Probability (1/year)",
            title="Age Spectrum, Mean Age: "
            + str(m_age)
            + ", Ratio of moments: "
            + str(rom),
        )
        ax.grid()

    return G
