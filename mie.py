"""This module contains functions to generate Mie scattering properties.

Python binding are implemented for the native SHDOM Mie module for computations of
monodisperse, i.e., single particle scattering properties. Supported particle types are
'water', 'aerosol' (in which case the index of refraction is user specified) or
'ice' (in which case index of refraction depends on wavelength).
The module also has additional support for integration over arbitrary particle distributions.

For water or ice particles the scattering properties may be averaged over the
desired spectral range with Planck function weighting. The phase functions in the
output scattering table are represented with Legendre series. For polarized output,
the six unique elements of the phase matrix are represented with Wigner d-function
expansions (Doicu et al., 2013, JQSRT, http://dx.doi.org/10.1016/j.jqsrt.2012.12.009).
"""
import os
import xarray as xr
import numpy as np

import miecode
import checks

def compute_table(wavelength, refractive_index,
                  minimum_effective_radius, max_integration_radius, verbose=True):
    """
    Parameters
    ----------
    wavelength: float
        The wavelength in microns.
    refractive_index: complex
        The refractive index should have a negative imaginary part. ri = n - ik
    minimum_effective_radius: float
        In microns, used to compute the minimum radius.
    max_integration_radius: float
        The largest radius to compute mie properties for in microns.
    verbose: bool
        True for progress prints from the fortran computations.

    Returns
    -------
    table: xr.Dataset
        A Dataset containing the scattering / extinction coefficients and
        tabulated Legendre coefficients per radius per wavelength.
        Each entry has 6 six Wigner d-function elements of the Mie phase matrix indexed by:
        P11, P22, P33, P44, P12, P34.
    """

    partype = 'A'
    avgflag = 'C'
    deltawave = -1

    wavelength_band = (wavelength, wavelength)

    # wavelength band
    wavelen1, wavelen2 = wavelength_band
    if wavelen1 > wavelen2:
        raise ValueError('wavelen1 must be <= wavelen2')

    wavelencen = miecode.get_center_wavelen(
        wavelen1=wavelen1,
        wavelen2=wavelen2
    )

    # set integration parameters
    if avgflag == 'A':
        xmax = 2 * np.pi * max_integration_radius / wavelen1
    else:
        xmax = 2 * np.pi * max_integration_radius / wavelencen
    maxleg = int(np.round(2.0 * (xmax + 4.0 * xmax ** 0.3334 + 2.0)))

    # set radius integration parameters
    nsize = miecode.get_nsize(
        sretab=minimum_effective_radius,
        maxradius=max_integration_radius,
        wavelen=wavelencen
    )

    radii = miecode.get_sizes(
        sretab=minimum_effective_radius,
        maxradius=max_integration_radius,
        wavelen=wavelencen,
        nsize=nsize
    )
    #compute mie properties
    extinct, scatter, nleg, legcoef, ierr, errmsg = \
        miecode.compute_mie_all_sizes(
            nsize=nsize,
            maxleg=maxleg,
            wavelen1=wavelen1,
            wavelen2=wavelen2,
            deltawave=deltawave,
            wavelencen=wavelencen,
            radii=radii,
            rindex=refractive_index,
            avgflag=avgflag,
            partype=partype,
            verbose=verbose
        )

    checks.check_errcode(ierr, errmsg)

    table = xr.Dataset(
        data_vars={
            'extinction': (['radius'], extinct),
            'scatter': (['radius'], scatter),
            'nleg': (['radius'], nleg),
            'legendre': (['stokes_index', 'legendre_index', 'radius'], legcoef)
            },
        coords={
            'radius': radii,
            'stokes_index': (['stokes_index'], ['P11', 'P22', 'P33', 'P44', 'P12', 'P34'])
            },
        attrs={
            'particle_type': 'Ice',
            'refractive_index': (refractive_index.real, refractive_index.imag),
            'refractive_index_source': 'refraction.py',
            'units': ['Radius [micron]', 'Wavelength [micron]'],
            'wavelength_band': wavelength_band,
            'wavelength_center': wavelencen,
            'wavelength_averaging': 'None',
            'wavelength_resolution': -1,
            'maximum_legendre': maxleg,
            'minimum_effective_radius':minimum_effective_radius,
            'maximum_integration_radius':max_integration_radius
            },
        )

    return table

def get_poly_table(size_distribution, mie_mono_table):
    """
    This methods calculates Mie scattering table for a polydisperse size distribution.
    For more information about the size_distribution see: lib/size_distribution.py.

    This function integrates the Mie table over radii for each entry in the
    size distribution Dataset. This dataset could be parameterized arbitrarily,
    however, a common use-case is according to effective radius and variace.

    Parameters
    ----------
    size_distribution: xr.Dataset
        A Dataset of number_density variable as a function of radius and the table parameterization.
        The shape of the dataset is ('radius', 'param1', 'param2',...'paramN').
        A common case is ('radius', 'reff', 'veff').
    mie_mono_table: xr.Dataset
        A Dataset of Mie legendre coefficients as a function of radius.
        See mie.get_mono_table function for more details.

    Returns
    -------
    poly_table: xr.Dataset
        A Dataset with the polydisperse Mie effective scattering properties:
        extinction, ssalb, legcoef. Each of these are a function of the parameterization
        defined by the size_distribution.

    Raises
    ------
    AssertionError
        If the mie_mono_table are not within the range of the size_distribution radii.

    Notes
    -----
    The radius in size_distribution is interpolated onto the mie_mono_table radii grid.
    This is to avoid interpolation of the Mie table coefficients.
    """
    checks.check_range(
        mie_mono_table,
        radius=(size_distribution.radius.min(), size_distribution.radius.max())
        )

    if (size_distribution.radius.size != mie_mono_table.radius.size) or \
            np.any(size_distribution.radius.data != mie_mono_table.radius.data):
        print('Warning: size_distribution radii differ to mie_mono_table radii. '
              'Interpolating the size distribution onto the Mie table grid.')
        size_distribution = size_distribution.interp(radius=mie_mono_table.radius)

    number_density = size_distribution['number_density'].values.reshape(
        (len(size_distribution['radius'])), -1
        )

    extinct, ssalb, nleg, legcoef = \
        miecode.get_poly_table(
            nd=number_density,
            ndist=number_density.shape[-1],
            nsize=mie_mono_table.coords['radius'].size,
            maxleg=mie_mono_table.attrs['maximum_legendre'],
            nleg1=mie_mono_table['nleg'],
            extinct1=mie_mono_table['extinction'],
            scatter1=mie_mono_table['scatter'],
            legcoef1=mie_mono_table['legendre'])

    grid_shape = size_distribution['number_density'].shape[1:]

    # all coords except radius
    coords = {name:coord for name, coord in size_distribution.coords.items()
              if name not in ('radius', 'stokes_index')}
    microphysics_names = list(coords.keys())
    coord_lengths = [np.arange(coord.size) for name, coord in coords.items()]
    legen_index = np.meshgrid(*coord_lengths, indexing='ij')

    table_index = np.ravel_multi_index(legen_index, dims=[coord.size for coord in coord_lengths])
    coords['table_index'] = (microphysics_names, table_index)
    coords['stokes_index'] = mie_mono_table.coords['stokes_index']

    poly_table = xr.Dataset(
        data_vars={
            'extinction': (microphysics_names, extinct.reshape(grid_shape)),
            'ssalb': (microphysics_names, ssalb.reshape(grid_shape)),
            'legcoef': (['stokes_index', 'legendre_index'] + microphysics_names,
                        legcoef.reshape(legcoef.shape[:2] + grid_shape)),},
        coords=coords
    )
    poly_table = poly_table.assign_attrs(size_distribution.attrs)
    poly_table = poly_table.assign_attrs(mie_mono_table.attrs)
    return poly_table
