import numpy as np
import bilby
import prediction_damp_dphase
from scipy.optimize import minimize
import module_correction

##################################
# Paths
data_path_train_damp = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_v4Tsurrnotidal_6D_150/deltaphase"
data_path_validation_damp = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_v4Tsurrnotidal_6D_150/deltaphase_validation"
data_path_train_dphase = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_v4Tsurrnotidal_6D_150/deltaphase"
data_path_validation_dphase = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_v4Tsurrnotidal_6D_150/deltaphase_validation"
data_path_train_damp = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_Hybrid_GPR_chieff_ltilde/falign_20_30_talign_2cycles/deltaphase"
data_path_validation_damp = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_Hybrid_GPR_chieff_ltilde/falign_20_30_talign_2cycles/deltaphase_validation"
data_path_train_dphase = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_Hybrid_GPR_chieff_ltilde/falign_20_30_talign_2cycles/deltaphase"
data_path_validation_dphase = "/home/luca/Projects/gpr_seobnrv4tsurrogate/dataset/hybrid/FD/v4Tsurr_Hybrid_GPR_chieff_ltilde/falign_20_30_talign_2cycles/deltaphase_validation"

model_path_damp = "/home/luca/Projects/gpr_seobnrv4tsurrogate/predictions/saved_data/test1/"
model_path_dphase = "/home/luca/Projects/gpr_seobnrv4tsurrogate/predictions/saved_data/test2/"
##########################################

duration = 32
sampling_frequency = 2048 
snr_network_target = 50 

# Output information
outdir = 'outdir_inj_v4TNRSurr_rec_v4TSurr'
label = 'inj_v4TNRSurr_rec_v4TSurr'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Random seed for result reproducibility
np.random.seed(88170235)

# Parameters for the injected signal
injection_parameters = dict(
        mass_1=1.4,
        mass_2=1.4,
        chi_1=0.00,
        chi_2=0.00,
        luminosity_distance=50.,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
        lambda_1=400,
        lambda_2=400)

# Arguments for injected and template signal
waveform_arguments = dict(
        waveform_approximant='SEOBNRv4T_Surrogate',
        reference_frequency=50.,
        minimum_frequency=20.)

wf_correction = prediction_damp_dphase.wf_correction(
        data_path_train_damp,
        data_path_validation_damp, 
        model_path_damp,
        data_path_train_dphase, 
        data_path_validation_dphase,
        model_path_dphase)

waveform_arguments['waveform_correction'] = wf_correction

# Injected waveform
waveform_generator_signal = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=module_correction.lal_binary_neutron_star_with_waveform_correction,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments)

# Template waveform
waveform_generator_template = bilby.gw.WaveformGenerator(
        duration=duration, 
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments)

# Set up interferometers (by default to their design sensitivity)
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
for ifo in ifos:
    ifo.minimum_frequency = 20

ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] + 2 - duration)
def snr_luminosity_distance(distance, snr_target):
    injection_parameters['luminosity_distance'] = distance
    ifos.inject_signal(
        waveform_generator=waveform_generator_signal,
        parameters=injection_parameters)
   
    snr_network = 0
    for ifo in ifos:
        snr_network += ifo.meta_data['optimal_SNR']**2
    snr_network = np.sqrt(snr_network)
    
    snr_diff = np.abs(snr_network-snr_target)
    return snr_diff


res = minimize(snr_luminosity_distance, 200, args=(snr_network_target), method="Nelder-Mead", tol=1e-6)
print('lum distance', res.x)
injection_parameters['luminosity_distance'] = float(res.x[0])

# Insert injected signal
ifos.inject_signal(
        waveform_generator=waveform_generator_signal,
        parameters=injection_parameters)

# Sample in Mc, symmetric mass ratio, lambda_tilde, delta_lambda, and chi_eff 
priors = bilby.gw.prior.BNSPriorDict()

for key in ['psi', 
            'geocent_time', 
            'ra', 
            'dec', 
            'chi_1', 
            'chi_2',
            'theta_jn', 
            'luminosity_distance', 
            'phase']:
    priors[key] = injection_parameters[key]

priors.pop('mass_1')
priors.pop('mass_2')
priors.pop('chirp_mass')
priors.pop('mass_ratio')

priors['mass_1'] = bilby.core.prior.Constraint(minimum=1, maximum=2, name='mass_1', latex_label='$m_1$', unit=None)
priors['mass_2'] = bilby.core.prior.Constraint(minimum=1, maximum=2, name='mass_2', latex_label='$m_2$', unit=None)
priors['chirp_mass'] = bilby.core.prior.Uniform(minimum=0.8, maximum=2, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
priors['mass_ratio'] = bilby.core.prior.Uniform(minimum=0.5, maximum=1, name='mass_ratio', latex_label='$q$', unit=None, boundary=None)
#priors['symmetric_mass_ratio'] = bilby.core.prior.Uniform(0.15, 0.25, name='symmetric_mass_ratio')
priors['lambda_1'] = bilby.core.prior.Constraint(minimum=0, maximum=5000, name='lambda_1', latex_label='$\\Lambda_1$', unit=None)
priors['lambda_2'] = bilby.core.prior.Constraint(minimum=0, maximum=5000, name='lambda_2', latex_label='$\\Lambda_2$', unit=None)
priors['lambda_tilde'] = bilby.core.prior.Uniform(0, 5000, name='lambda_tilde')
priors['delta_lambda'] = bilby.core.prior.Uniform(-5000, 5000, name='delta_lambda')

# Setup likelihood using ifos data and templete waveforms to infer injected signal
likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator_template,
        priors=priors,
        time_marginalization=False, 
        phase_marginalization=False,
        distance_marginalization=False)

# Run
sampler = 'dynesty'
npoints = 500
maxmcmc = 5000
np.seterr(divide='ignore')
result = bilby.run_sampler(
        likelihood=likelihood, 
        priors=priors,
        sampler=sampler,
        npoints=npoints,
        injection_parameters=injection_parameters, 
        outdir=outdir, 
        label=label,
        maxmcmc=maxmcmc,
        conversion_function=bilby.gw.conversion.generate_all_bns_parameters)

# Plot results
result.plot_corner()
