"""
Contains:

Calculation of the index of refraction of ice over the microwave frequency range including temperature dependence.
"""

import numpy as np

def ice_index_of_refraction(mu, T):
    """
    Calculates the index of refraction of pure ice in the microwave range.

    If `mu` or `T` are specified with unequal length then the index of refraction
    will be calculated for all combinations

    Parameters
    ----------
    mu : float / np.ndarray
        The Frequency in Giga-Hertz (GHz). Valid range is 0.01 - 1000.0
    T : float / np.ndarray
        Temperature in Kelvin. Valid range is 243.0 - 273.0

    Returns
    -------
    index_real : np.ndarray, float
        The real component of the index of refraction
    index_imaj : np.ndarray, float
        The imaginary component of the index of refraction

    Notes
    -----
    The range is limited by the validity of the real component of the
    dielectric constant `real_dielectric_ice`.

    See Also
    --------
    `imaginary_dielectric_ice`
    `real_dielectric_ice`
    """
    mu = np.atleast_1d(mu)
    T = np.atleast_1d(T)

    # Allow broadcasting.
    if (mu.ndim == 1) & (T.ndim == 1) & (mu.size != T.size):
        T = T[:,None]

    E_real = real_dielectric_ice(mu, T)
    E_imaginary = imaginary_dielectric_ice(mu, T)

    index_of_refraction = np.sqrt(E_real + 1j*E_imaginary)

    index_real = index_of_refraction.real
    index_imaginary = index_of_refraction.imag

    return index_real, index_imaginary


def real_dielectric_ice(mu,T):
    """
    Calculates the real component of the dielectric constant of pure ice.

    Parameters
    ----------
    mu : float / np.ndarray
        The Frequency in Giga-Hertz (GHz). Valid range is 0.01 - 1000.0
    T : float / np.ndarray
        Temperature in Kelvin. Valid range is 243.0 - 273.0

    Returns
    -------
    E_real : np.ndarray
        The real component of the dielectric constant.

    Raises
    ------
    ValueError
        If inputs are outside valid ranges.

    References
    ----------
    These calculations follow

    Mätzler, C.: Thermal microwave radiation: applications for remote
    sensing, vol. 52 of IET electromagnetic waves series, Institution
    of Engineering and Technology, 2006

    Upon the recommendation of

    Eriksson, P., Ekelund, R., Mendrok, J., Brath, M., Lemke, O., and Buehler, S. A.:
    A general database of hydrometeor single scattering properties at microwave and
    sub-millimetre wavelengths, Earth Syst. Sci. Data, 10, 1301–1326,
    https://doi.org/10.5194/essd-10-1301-2018, 2018

    the frequency range is extended 1000 GHz
    """
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)

    if np.any(T < 243.0) or np.any(T > 273.0):
        raise ValueError(
            "Temperature Values outside of the Valid Range (243 K - 273 K) were Detected."
        )
    if (np.any(mu < 0.01) or np.any(mu > 1000.0)):
        raise ValueError(
            "Frequency Values outside of the Valid Range (0.01 GHz - 1000 GHz) were Detected."
        )

    E_real = 3.1884 + 9.1e-4 * (T - 273)
    return E_real

def imaginary_dielectric_ice(mu,T):
    """
    Calculates the imaginary component of the dielectric constant of pure ice.

    Parameters
    ----------
    mu : float / np.ndarray
        The Frequency in Giga-Hertz (GHz).
    T : float / np.ndarray
        Temperature in Kelvin
        
    Returns
    -------
    E_imaginary : np.ndarray
        The imaginary component of the dielectric constant.

    Notes
    -----
    No bounds are provided on the inputs but errors in the fit vary.

    Error / uncertainty is approximately 0.45e-5 * mu

    Relative Errors are
    5% at 270K to 7.5% at 250 K to 14% at 200 K
    essentially independent of frequency for freqiencies greater than 3 GHz.


    References
    ----------
    These calculations follow

    Mätzler, C.: Thermal microwave radiation: applications for remote
    sensing, vol. 52 of IET electromagnetic waves series, Institution
    of Engineering and Technology, 2006

    which is the recommendation of

    Eriksson, P., Ekelund, R., Mendrok, J., Brath, M., Lemke, O., and Buehler, S. A.:
    A general database of hydrometeor single scattering properties at microwave and
    sub-millimetre wavelengths, Earth Syst. Sci. Data, 10, 1301–1326,
    https://doi.org/10.5194/essd-10-1301-2018, 2018
    """
    mu = np.atleast_1d(mu)
    T = np.atleast_1d(T)

    theta = 300.0/T - 1.0 #in K
    alpha = (0.00504 + 0.0062*theta)*np.exp(-22.1 * theta) # in GHz

    # Low temperature  term
    B1 = 0.0207 # K/GHz
    b = 335.0 #K
    B2 = 1.16e-11 #1/GHz**3
    beta_m = (B1/T) * np.exp(b/T)/((np.exp(b/T) -1.0)**2) + B2*mu**2

    # higher temperature correction
    delta_beta = np.exp(-9.963 + 0.0372*(T - 273.16))

    beta = beta_m + delta_beta

    E_imaginary = alpha / mu + beta * mu

    return E_imaginary
