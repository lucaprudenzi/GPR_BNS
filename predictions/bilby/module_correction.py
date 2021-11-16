from bilby.gw.source import _base_lal_cbc_fd_waveform
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

def lal_binary_neutron_star_with_waveform_correction(
        frequency_array, 
        mass_1, 
        mass_2, 
        luminosity_distance, 
        a_1, 
        tilt_1,
        phi_12, 
        a_2, 
        tilt_2, 
        phi_jl,
        theta_jn, 
        phase, 
        lambda_1, 
        lambda_2,
        **kwargs):

    """ A Binary Neutron Star waveform model using lalsimulation
    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    kwargs: dict
        Optional keyword arguments
        Supported arguments:
        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 a\nd 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.
    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """     

    waveform_kwargs = dict(
            waveform_approximant='SEOBNRv4T_Surrogate', 
            reference_frequency=50.0,
            minimum_frequency=20.0, 
            maximum_frequency=frequency_array[-1],
            waveform_error_model=None, 
            catch_waveform_errors=True, 
            pn_spin_order=-1,
            pn_tidal_order=-1,
            pn_phase_order=-1,
            pn_amplitude_order=0)
    
    waveform_kwargs.update(kwargs)
    
    # Sanity check
    if waveform_kwargs['waveform_correction'] is None:
        raise ValueError('waveform_correction_model not specified.')
    else:
        wf_correction = waveform_kwargs['waveform_correction']
    
    if waveform_kwargs['waveform_approximant'] != 'SEOBNRv4T_Surrogate':
        raise ValueError('Waveform uncertainty model is only available for SEOBNRv4T_Surrogate.')

    # Compute waveform polarizations 
    # Return: polarizations = {'plus': h_plus, 'cross': h_cross}
    polarizations = _base_lal_cbc_fd_waveform(
            frequency_array=frequency_array, 
            mass_1=mass_1, 
            mass_2=mass_2,
            luminosity_distance=luminosity_distance,
            theta_jn=theta_jn,
            phase=phase,
            a_1=a_1,
            a_2=a_2,
            tilt_1=tilt_1,
            tilt_2=tilt_2,
            phi_12=phi_12,
            phi_jl=phi_jl,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            **waveform_kwargs)
    
    chi_1z = a_1 * np.cos(tilt_1)
    chi_2z = a_2 * np.cos(tilt_2)
    
    # Obtain damp and dphi at the frequency nodes freq
    damp, dphi, freq = wf_correction.draw_sample_damp_dphase(
            mass_1,
            mass_2,
            chi_1z, 
            chi_2z, 
            lambda_1, 
            lambda_2)

    # Interpolant to connect the frequency nodes
    # ext=0 (spline extrapolation) or ext=3 (use boundary value)
    damp_interp = InterpolatedUnivariateSpline(freq, damp, ext=3)
    dphase_interp= InterpolatedUnivariateSpline(freq, dphi, ext=3)
   
    f = frequency_array
    idx = np.where((f >= waveform_kwargs['minimum_frequency']) &
                   (f <= waveform_kwargs['maximum_frequency']))
    
    wf_corrections = (1 + damp_interp(f[idx]))*np.exp(1j*dphase_interp(f[idx]))
    
    # Apply amplitude and phase error model
    for key, pol in polarizations.items():
        # print(len(pol), len(wf_fac), len(f[idx]))
        pol[idx] *= wf_corrections
    
    return polarizations

