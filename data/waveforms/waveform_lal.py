from waveform_base import waveform_base
import lal
import numpy as np
import lalsimulation as lalsim
import pathlib
import os
import copy
import json
import scipy as scp
import matplotlib.pyplot as plt
import pycbc
import compare_waveforms_base as compare_wf
from waveform_base import waveform_base


class waveform_lal(waveform_base):
    def __init__(self, fixed_par=None, approx_fd="SEOBNRv4T_Surrogate", approx_td = "SEOBNRv4T", f_lower=30, df = 0.01):
        super().__init__()

        # Starting frequency for generation of the FD waveform
        self.f_lower = f_lower
        self.f_upper = 8192
        # NOTE: df=0.001 for f_lower = 15 needed to avoid artifacts for 
        # the phase in FD
        self.df = df

        self.freq_interp = np.linspace(self.f_lower, self.f_upper, 100000) 
        self.srate = self.f_upper*2
        self.delta_t = 1/self.srate
        self.distance_Mpc = 1 
        self.distance = self.distance_Mpc * lal.PC_SI * 1e6
        self.inclination = 0.
        self.coa_phase = 0.
        self.ecc = 0. 

        self.par = {}
        self.approx_fd = approx_fd
        self.approx_td = approx_td
        
        if fixed_par is not None:
            for key, value in fixed_par.items():
                self.par[key] = value


    def get_h_td(self, method="lal"):

        if method == "lal": 
            ma = lalsim.SimInspiralCreateModeArray()
            params_lal = lal.CreateDict()
            lalsim.SimInspiralModeArrayActivateMode(ma, 2, 2)
            lalsim.SimInspiralWaveformParamsInsertModeArray(params_lal, ma)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params_lal, self.par['lambda1'])
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params_lal, self.par['lambda2'])
            approx_lal = lalsim.GetApproximantFromString(self.approx_td)
            self.f_lower = 200

            hp_td, hc_td = lalsim.SimInspiralChooseTDWaveform(
                    self.par['m1']*lal.MSUN_SI, 
                    self.par['m2']*lal.MSUN_SI,
                    0,
                    0,
                    self.par['s1z'],
                    0,
                    0,
                    self.par['s2z'],
                    self.distance, 
                    self.inclination,
                    self.coa_phase,
                    0.,
                    self.ecc,
                    0., 
                    self.delta_t,
                    self.f_lower,
                    self.f_lower,
                    params_lal,
                    approx_lal)

            self.hp_td = hp_td.data.data
            self.hc_td = hc_td.data.data
            self.h_td_to_amp_phase_freq_td()
            #self.time = (np.arange(self.hp_td.data.length)-np.argmax(hp_td.data.data))*hp_td.deltaT
            self.time = (np.arange(int(len(self.amp_td)))-np.argmax(self.amp_td))*hp_td.deltaT
            self.dt = np.diff(self.time)[0]

        if method == "pycbc": 
            self.hp_td, self.hc_td = pycbc.waveform.get_td_waveform(
                mass1=self.par['m1'], 
                mass2=self.par['m2'],
                spin1x=0,
                spin1y=0,
                spin1z=self.par['s1z'],
                spin2x=0,
                spin2y=0,
                spin2z=self.par['s2z'],
                eccentricity=self.ecc,
                distance=self.distance_Mpc,
                f_lower=self.f_lower,
                delta_t=self.delta_t,
                lambda1=self.par['lambda1'],
                lambda2=self.par['lambda2'],
                approximant=self.approx_td)
            
            self.time = hp_td.get_sample_times()
        
        self.dt = np.diff(self.time)[0] 
        self.h_td_to_amp_phase_freq_td() 

    def get_h_fd(self, method="lal"):
        """
        FD hp and hc from LAL
        """
        #self.compute_df() 
         
        if method == "lal":
            params_lal = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params_lal, self.par['lambda1'])
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params_lal, self.par['lambda2'])
            approx_lal = lalsim.GetApproximantFromString(self.approx_fd)
            
            hp_fd, hc_fd = lalsim.SimInspiralChooseFDWaveform(
                    self.par['m1']*lal.MSUN_SI,
                    self.par['m2']*lal.MSUN_SI,
                    0,
                    0, 
                    self.par['s1z'],
                    0,
                    0,
                    self.par['s2z'],
                    self.distance,
                    self.inclination,
                    self.coa_phase,
                    0,
                    self.ecc,
                    0,
                    self.df,
                    self.f_lower,
                    self.f_upper,
                    self.f_lower,
                    params_lal,
                    approx_lal)
            
            self.freq = np.arange(hp_fd.data.length)*self.df

            self.hp_fd = hp_fd.data.data
            self.hc_fd = hc_fd.data.data
        
        if method == "pycbc": 
            self.hp_fd, self.hc_fd = pycbc.waveform.get_fd_waveform(
                mass1=self.par['m1'],
                mass2=self.par['m2'],
                spin1x=0,
                spin1y=0,
                spin1z=self.par['s1z'],
                spin2x=0,
                spin2y=0,
                spin2z=self.par['s2z'],
                eccentricity=self.ecc,
                distance=self.distance_Mpc,
                delta_f=self.df,
                f_lower=self.f_lower,
                f_final=self.f_upper,
                lambda1=self.par['lambda1'],
                lambda2=self.par['lambda2'],
                approximant=self.approx_fd)

            self.freq = self.hp_fd.sample_frequencies
        
        self.h_fd_to_amp_phase_fd() 

    def get_h_td_from_fd(self):
        self.hp_td, self.time = self.h_fd_ifft(self.hp_fd, self.freq, self.f_lower, self.f_upper)
        self.hc_td, self.time = self.h_fd_ifft(self.hc_fd, self.freq, self.f_lower, self.f_upper)
        self.dt = np.diff(self.time)[0] 
        self.h_td_to_amp_phase_freq_td() 

if __name__=="__main__":
    import matplotlib.pyplot as plt
    par={}
    par['m1']=1.4
    par['m2']=1.4
    par['s1z']=0
    par['s2z']=0

    par['lambda1'] = 400
    par['lambda2'] = 400
    wf_lal = waveform_lal(par)
    wf_lal.get_h_fd()


    plt.plot(wf_lal.freq, np.real(wf_lal.hp_fd))
    plt.plot(wf_lal.freq, np.imag(wf_lal.hp_fd))
    plt.xlim(0,2000)
    plt.savefig('a.png')