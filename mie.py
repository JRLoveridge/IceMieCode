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
import typing
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


def get_phase_function(legcoef, angles, phase_elements='All'):
    """Calculates phase function from legendre tables.

    If a multi-dimensional table is passed then phase functions for all
    microphysical dimensions (including individual radii if available)
    will be sampled at the specified `angles`.

    Parameters
    ----------
    legcoef : xr.DataArray
        Contains the legendre/Wigner coefficients for the phase function.
        should be produced by mie.get_mie_mono or  mie.get_poly_table or
        in the same format.
    angles : array_like of floats
        scattering angles to sample the phase function at in degrees.
        should be a 1D array_like
    phase_elements : str, list/tuple of strings
        valid values are from P11, P22, P33, P44, P12, P34, or All.

    Returns
    -------
    phase_array : xr.DataArray
        Contains the phase function at the sampled `angles` for each of the provided
        legendre/Wigner series, for the specified `phase_elements`

    Raises
    ------
    ValueError
        If `phase_elements` is not composed of valid strings
    TypeError
        If `phase_elements` is not of type ``str``, ``tuple`` or ``list``.

    See Also
    --------
    mie.get_poly_table
    mie.get_mono_table

    Example
    -------
    >>> legcoef = mie_mono_table.legendre[:,:,50:55]
    #select 5 radii from the mie_mono_table.

    >>> phase_array = get_phase_function(legcoef,
                                         np.linspace(0.0,180.0,361),
                                         phase_elements='All')
    >>> phase_array
    <xarray.DataArray 'phase_function' (phase_elements: 6, scattering_angle: 361, radius: 5)>
    array([[[ 5.06317383e-03,  6.17436739e-03,  7.50618055e-03,
              9.09753796e-03,  1.09932469e-02],
            [ 5.06292842e-03,  6.17406424e-03,  7.50580709e-03,
              9.09707788e-03,  1.09926835e-02],
            [ 5.06219361e-03,  6.17315574e-03,  7.50468718e-03,
              9.09570046e-03,  1.09909941e-02],
            ...,
            [ 2.58962740e-03,  3.01506580e-03,  3.48956930e-03,
              4.01409063e-03,  4.58831480e-03],
            [ 2.58983485e-03,  3.01530003e-03,  3.48983146e-03,
              4.01438121e-03,  4.58863331e-03],
            [ 2.58990400e-03,  3.01537826e-03,  3.48991877e-03,
              4.01447807e-03,  4.58873948e-03]],

           [[ 5.06317383e-03,  6.17436739e-03,  7.50618055e-03,
              9.09753796e-03,  1.09932479e-02],
            [ 5.06292889e-03,  6.17406424e-03,  7.50580709e-03,
              9.09707882e-03,  1.09926844e-02],
            [ 5.06219361e-03,  6.17315574e-03,  7.50468718e-03,
              9.09570139e-03,  1.09909950e-02],
    ...
            [-4.16639125e-07, -4.87358477e-07, -5.66944379e-07,
             -6.55818667e-07, -7.54250607e-07],
            [-1.04164208e-07, -1.21844508e-07, -1.41741438e-07,
             -1.63960422e-07, -1.88568734e-07],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              0.00000000e+00,  0.00000000e+00]],

           [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              0.00000000e+00,  0.00000000e+00],
            [ 4.72140493e-10,  6.52067178e-10,  8.93253749e-10,
              1.21347665e-09,  1.63433822e-09],
            [ 1.88840676e-09,  2.60805422e-09,  3.57272101e-09,
              4.85350737e-09,  6.53681553e-09],
            ...,
            [ 1.69794856e-09,  2.33928010e-09,  3.19921267e-09,
              4.34327951e-09,  5.85340043e-09],
            [ 4.24516838e-10,  5.84860882e-10,  7.99858901e-10,
              1.08589548e-09,  1.46345203e-09],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              0.00000000e+00,  0.00000000e+00]]], dtype=float32)
    Coordinates:
      * phase_elements    (phase_elements) <U3 'P11' 'P22' 'P33' 'P44' 'P12' 'P34'
      * scattering_angle  (scattering_angle) float64 0.0 0.5 1.0 ... 179.5 180.0
      * radius            (radius) float32 0.1186 0.1224 0.1263 0.1303 0.1343
    """
    pelem_dict = {'P11': 1, 'P22': 2, 'P33': 3, 'P44': 4, 'P12': 5, 'P34': 6}
    if phase_elements == 'All':
        phase_elements = list(pelem_dict.keys())
    elif isinstance(phase_elements, (typing.List, typing.Tuple)):
        for element in phase_elements:
            if element not in pelem_dict:
                raise ValueError("Invalid value for phase_elements '{}' "
                                 "Valid values are '{}'".format(element, pelem_dict.keys()))
    elif phase_elements in pelem_dict:
        phase_elements = [phase_elements]
    else:
        raise TypeError("phase_elements argument should be either 'All' or a list/tuple of strings"
                        "from {}".format(pelem_dict.keys()))

    coord_sizes = {name:legcoef[name].size for name in legcoef.coords
                   if name not in ('stokes_index', 'legendre_index', 'table_index')}

    coord_arrays = [np.arange(size) for size in coord_sizes.values()]
    coord_indices = np.meshgrid(*coord_arrays, indexing='ij')
    flattened_coord_indices = [coord.ravel() for coord in coord_indices]

    phase_functions_full = []

    loop_max = 1
    if flattened_coord_indices:
        loop_max = flattened_coord_indices[0].shape[0]

    for i in range(loop_max):
        index = tuple([slice(0, legcoef.stokes_index.size),
                       slice(0, legcoef.legendre_index.size)] +
                      [coord[i] for coord, size in zip(flattened_coord_indices, coord_sizes.values())
                       if size > 1])
        single_legcoef = legcoef.data[index]

        phase_functions = []
        for phase_element in phase_elements:
            pelem = pelem_dict[phase_element]

            phase = miecode.transform_leg_to_phase(
                maxleg=legcoef.legendre_index.size - 1,
                nphasepol=6,
                legcoef=single_legcoef,
                pelem=pelem,
                nleg=legcoef.legendre_index.size - 1,
                nangle=len(angles),
                angle=angles
            )
            phase_functions.append(phase)
        phase_functions_full.append(np.stack(phase_functions, axis=0))

    small_coord_sizes = {name:size for name, size in coord_sizes.items() if size > 1}
    coords = {
        'phase_elements': np.array(phase_elements),
        'scattering_angle': angles
        }
    for name in coord_sizes:
        coords[name] = legcoef.coords[name]

    phase_functions_full = np.stack(phase_functions_full, axis=-1).reshape(
        [len(phase_elements), len(angles)] + list(small_coord_sizes.values())
    )

    phase_array = xr.DataArray(
        name='phase_function',
        dims=['phase_elements', 'scattering_angle'] + list(small_coord_sizes.keys()),
        data=phase_functions_full,
        coords=coords
    )
    return phase_array
