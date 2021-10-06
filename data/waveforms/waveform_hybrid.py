#!/usr/bin/env python3
import pylab 
from pylab import arange,pi,sin,cos,sqrt
from waveform_base import waveform_base
from waveform_lal import waveform_lal
from waveform_nr import waveform_nr
import compare_waveforms_base as compare_wf
import numpy as np
import lal
from pycbc.types import TimeSeries, FrequencySeries
import pycbc.waveform as wf
import matplotlib.pyplot as plt
import compare_waveforms_base as compare_wf
import glob
import json
import copy
import os

class waveform_hybrid(waveform_base):
    def __init__(self):
        super().__init__()
        
        self.f_start_window = 35
        self.f_end_window = 45
        self.N_cycles = 7
    
    def get_h_td(self, wf_lal, wf_nr):
        #wf_lal = copy.deepcopy(wf_lal_copy)
        self.par = wf_lal.par
        
        print("HYB construction v4Tsurr-NR")
        self.factor = (wf_nr.par['m1']+wf_nr.par['m2'])*lal.G_SI*lal.MSUN_SI/lal.C_SI**3    
        # window for the TD alignment of the phases
        self.hyb_window_start = wf_nr.time[0]+self.t_align_start*self.factor
        hyb_window_width = self.N_cycles/(wf_nr.f_lower*self.factor)
        t_align_stop = self.t_align_start+hyb_window_width
        self.hyb_window_end = wf_nr.time[0]+t_align_stop*self.factor

        self.h_aligned_TD(wf_lal, wf_nr)
     
        # Times choosen with equal sampling in the 3 regions
        time_pre = wf_lal.time[wf_lal.time<self.hyb_window_start]
        time_in = np.arange(time_pre[-1]+np.diff(wf_lal.time)[0], self.hyb_window_end, np.diff(wf_lal.time)[0])
        time_post = np.arange(time_in[-1]+np.diff(wf_lal.time)[0], wf_nr.time[-1], np.diff(wf_lal.time)[0])
        time_pad = np.arange(time_post[-1]+np.diff(wf_lal.time)[0], wf_lal.time[-1], np.diff(wf_lal.time)[0])
        self.time = np.concatenate([time_pre, time_in, time_post, time_pad])

        """ Pre window"""
        hp_td_pre = wf_lal.hp_td[wf_lal.time<self.hyb_window_start]
        hc_td_pre = wf_lal.hc_td[wf_lal.time<self.hyb_window_start]

        """ In window: interpolate to combine the 2 phases"""
        hp_td_in_nr = compare_wf.interpolate_grid(wf_nr.time, wf_nr.hp_td, time_in)
        hc_td_in_nr = compare_wf.interpolate_grid(wf_nr.time, wf_nr.hc_td, time_in)
        hp_td_in_lal = compare_wf.interpolate_grid(wf_lal.time, wf_lal.hp_td, time_in)
        hc_td_in_lal = compare_wf.interpolate_grid(wf_lal.time, wf_lal.hc_td, time_in)

        hp_td_in = hp_td_in_lal*\
                (1-self.hybridization_window(time_in, time_in[0], time_in[-1]))+\
                hp_td_in_nr*self.hybridization_window(time_in, time_in[0], time_in[-1])
        hc_td_in = hc_td_in_lal*\
                (1-self.hybridization_window(time_in, time_in[0], time_in[-1]))+\
                hc_td_in_nr*self.hybridization_window(time_in, time_in[0], time_in[-1])

        """ Post window"""
        hp_td_post = compare_wf.interpolate_grid(wf_nr.time, wf_nr.hp_td, time_post)
        hc_td_post = compare_wf.interpolate_grid(wf_nr.time, wf_nr.hc_td, time_post)

        """ Combine 3 regions"""
        hp_td = np.concatenate([hp_td_pre, hp_td_in, hp_td_post])
        hc_td = np.concatenate([hc_td_pre, hc_td_in, hc_td_post])
        hp_td = np.pad(hp_td, (0,len(self.time)-len(hp_td)),'constant')
        hc_td = np.pad(hc_td, (0,len(self.time)-len(hc_td)),'constant')
        
        self.hp_td = TimeSeries(hp_td, delta_t = np.diff(self.time)[0])
        self.hc_td = TimeSeries(hc_td, delta_t = np.diff(self.time)[0])

        self.freq_td = wf.utils.frequency_from_polarizations(self.hp_td, self.hc_td)
        self.phase_td = wf.utils.phase_from_polarizations(self.hp_td, self.hc_td, remove_start_phase=False)

        """ Tapeing the post merger part from NR """
        t = self.time[:-1]
        t_for_tap = t[t>-10]            # take only the last section of the data to compute 
        f_for_tap = self.freq_td[t>-10] # time-freq relation 

        self.tap_window_start = t_for_tap[f_for_tap>wf_nr.f_merg][0]
        self.tap_window_end = t_for_tap[f_for_tap>1.2*wf_nr.f_merg][0]
        self.tap_window = self.PlanckTaper(self.time, self.tap_window_start, self.tap_window_end, side="tap_at_right")

        #self.tap_window_center = np.abs(self.tap_window_end+self.tap_window_start)/2.
        #self.tap_window_width = np.abs(self.tap_window_end-self.tap_window_start)
        #self.tap_window = self.TanhTaper(self.time, self.tap_window_center, self.tap_window_width, side="tap_at_right") # non zero at the end of the window
        
        #self.hp_td = np.array(self.hp_td)
        #self.hc_td = np.array(self.hc_td)
        self.hp_td = np.array(self.hp_td*self.tap_window)
        self.hc_td = np.array(self.hc_td*self.tap_window)
        
        index_hpminus = np.argwhere(self.hp_td==-0)
        index_hcminus = np.argwhere(self.hc_td==-0)
        self.hp_td[index_hpminus] = 0 
        self.hc_td[index_hpminus] = 0 

        self.hp_td = TimeSeries(self.hp_td, delta_t = np.diff(self.time)[0])
        self.hc_td = TimeSeries(self.hc_td, delta_t = np.diff(self.time)[0])

        #self.time = self.time-self.time[np.argmax(np.abs(self.hp_td-1j*self.hc_td))] 
        self.amp_td = np.abs(self.hp_td+1j*self.hc_td)
        self.phase_td = wf.utils.phase_from_polarizations(self.hp_td, self.hc_td, remove_start_phase=False)
        self.freq_td = wf.utils.frequency_from_polarizations(self.hp_td, self.hc_td)
        self.f_nr_start = f_for_tap[t_for_tap>self.hyb_window_start][0]
        self.f_nr_start_full = f_for_tap[t_for_tap>self.hyb_window_end][0]

    def h_aligned_TD(self, wf_lal, wf_nr):
        """ 
        The returned phase is the shifted one
        """

        wf_lal.phase_td = compare_wf.phaseshift_td(wf_nr.time, wf_nr.phase_td, wf_lal.time, wf_lal.phase_td, self.hyb_window_start, self.hyb_window_end)
        wf_lal.hp_td = wf_lal.amp_td*np.cos(wf_lal.phase_td) # phase = unwrap(arctan(hc/hp))
        wf_lal.hc_td = wf_lal.amp_td*np.sin(wf_lal.phase_td) # phase = unwrap(arctan(tan(p))

    def hybridization_window(self, time, time_i, time_f):
        """
        Activation function: 0 before time_i, 1 after time_f
        """
        res = 0.5*(1-np.cos(np.pi*(time-time_i)/(time_f-time_i)))
        return res
      
    def get_h_fd_from_td(self, wf_lal):
        self.hp_fd, self.freq = self.h_td_fft(self.hp_td, self.time)
        self.hc_fd, self.freq = self.h_td_fft(self.hc_td, self.time)
       
        self.df = np.diff(self.freq)[0]
        self.h_fd_to_amp_phase_fd()

        self.phase_fd = compare_wf.phaseshift_fd(self.freq, self.phase_fd, wf_lal.freq, wf_lal.phase_fd, self.f_start_window, self.f_end_window)


if __name__== "__main__":


    yes_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_yes.json"))
    maybe_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_maybe.json"))
    short_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_short.json"))
    lambda5000_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_lambda5000.json"))
    sacra_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_sacra.json"))
    thc_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_thc.json"))
    doubts_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_doubts.json"))
    
    used_dict = {}
    used_dict.update(sacra_dict)
    used_dict.update(maybe_dict)
    used_dict.update(yes_dict)
    #used_dict.update(doubts_dict)
    #used_dict.update(thc_dict)
    #used_dict.update(short_dict)

    basedir_phase = "../NR_data/BAM_SACRA_data_deltaphase/"
    basedir_amp = "../NR_data/BAM_SACRA_data_deltaamp/"
    values_spin1 = [] 
    values_spin2 = [] 
    values_chieff = []
    values_mass1 = [] 
    values_mass2 = [] 
    values_Mc = []
    values_q = []
    values_M = []
    values_lambda1 = [] 
    values_lambda2 = [] 
    
    # save all the parameters of all the sets
    for key, value in used_dict.items(): 
        values_spin1.append(value['s1z']) 
        values_spin2.append(value['s2z']) 
        values_mass1.append(value['m1']) 
        values_mass2.append(value['m2']) 
        values_lambda1.append(value['lambda1']) 
        values_lambda2.append(value['lambda2']) 
    
    # cycle only on sets in used_dict
    for key, value in used_dict.items():
        if os.path.exists(basedir_phase+"placeholder/"+str(key[:-3])):
            continue
        if not os.path.exists(basedir_phase+"placeholder/"+str(key[:-3])):
            os.makedirs(basedir_phase+"placeholder/"+str(key[:-3]))

        print("Load NR wf")
        wf_nr = waveform_nr() 
        wf_nr.set_parameters(key, value)

        wf_nr.get_h_td()
        wf_nr.get_h_fd_from_td()
        
        print("Load LAL wf")
        par_bns = copy.deepcopy(wf_nr.par)
        wf_bns = waveform_lal(par_bns, approx_fd="SEOBNRv4T_Surrogate")
        wf_bns.get_h_fd()
        wf_bns.get_h_td_from_fd()
        
        print("Load Hybrid wf")
        wf_hyb = waveform_hybrid()
        wf_hyb.t_align_start = value["t_align_start"] 
        wf_hyb.get_h_td(wf_bns, wf_nr)
        wf_hyb.get_h_fd_from_td(wf_bns)
        
        #shift_centering = wf_hyb.tap_window_start
        #wf_hyb.time = wf_hyb.time - shift_centering
        #wf_bns.time = wf_bns.time - shift_centering
        #wf_nr.time = wf_nr.time - shift_centering

        #wf_hyb.hyb_window_start = wf_hyb.hyb_window_start - shift_centering
        #wf_hyb.hyb_window_end = wf_hyb.hyb_window_end - shift_centering
        #wf_hyb.tap_window_end = wf_hyb.tap_window_end  - shift_centering
        #wf_hyb.tap_window_start = wf_hyb.tap_window_start - shift_centering

        fig_width_pt = 420.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = 2.8*fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        
        params = {'backend': 'pdf',
                   'axes.labelsize': 11,
                   'legend.fontsize': 9,
                   'xtick.labelsize': 11,
                   'ytick.labelsize': 11,
                   'text.usetex': True,
                   'figure.figsize': fig_size}
        
        pylab.rcParams.update(params)

        """
        TIME DOMAIN PLOT
        """
        
        # WF PLOT
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 3)
        #fig.suptitle(key.decode('ascii'), fontsize=16)
        ax1 = fig.add_subplot(gs[0, :]) 
        ax1.plot(wf_hyb.time, wf_hyb.hp_td, "--", label='hyb')
        ax1.plot(wf_bns.time, wf_bns.hp_td, label='lal')
        ax1.plot(wf_nr.time, wf_nr.hp_td, alpha=0.3, label='nr')
        
        ax1_1 = ax1.twinx() 
        ax1_1.plot(wf_hyb.time[:-1], wf_hyb.freq_td, label='nr')
        ax1_1.set_ylim(-max(wf_hyb.freq_td),max(wf_hyb.freq_td))
        
        ax1.axvline(wf_hyb.hyb_window_start, color='black')
        ax1.axvline(wf_hyb.hyb_window_end, color='blue')
        ax1.axvline(wf_hyb.tap_window_start, color='green')
        ax1.axvline(wf_hyb.tap_window_end, color='red')
        ax1.set_xlim(left=wf_nr.time[0], right=wf_nr.time[-1]) 
        ax1.set_xlabel(r't[s]')
        ax1.set_ylabel(r'$h$')
        ax1.legend()

        ax1.axvline(wf_hyb.hyb_window_start)
        ax1.axvline(wf_hyb.hyb_window_end)
        ax1.axvline(wf_hyb.tap_window_start)
        ax1.axvline(wf_hyb.tap_window_end)
        #ax1.legend()
        
        # PHASE PLOT
        p1 = compare_wf.interpolate_grid(wf_bns.time, wf_bns.phase_td, wf_hyb.time)
        p2 = compare_wf.interpolate_grid(wf_hyb.time, wf_hyb.phase_td, wf_hyb.time)
        delta_phase_td = p1-p2
        delta_phase_td = delta_phase_td - delta_phase_td[0]
        
        ax2 = fig.add_subplot(gs[1, :]) 
        ax2.plot(wf_hyb.time, delta_phase_td)
        ax2.axvline(wf_hyb.hyb_window_start, color='black')
        ax2.axvline(wf_hyb.hyb_window_end, color='blue')
        ax2.axvline(wf_hyb.tap_window_start, color='green')
        ax2.axvline(wf_hyb.tap_window_end, color='red')
        ax2.set_xlim(left=wf_nr.time[0], right=wf_hyb.tap_window_start)
        ax2.set_ylim(-6,6)
        ax2.set_xlabel(r't[s]')
        ax2.set_ylabel(r'$\Delta\phi$')
        
        a1 = compare_wf.interpolate_grid(wf_bns.time, wf_bns.amp_td, wf_hyb.time)
        a2 = compare_wf.interpolate_grid(wf_hyb.time, wf_hyb.amp_td, wf_hyb.time)
        delta_amp_td = a1/a2
        
        ax3 = fig.add_subplot(gs[2, :]) 
        ax3.plot(wf_hyb.time, delta_amp_td)
        ax3.axvline(wf_hyb.hyb_window_start, color='black')
        ax3.axvline(wf_hyb.hyb_window_end, color='blue')
        ax3.axvline(wf_hyb.tap_window_start, color='green')
        ax3.axvline(wf_hyb.tap_window_end, color='red')
        ax3.set_xlim(left=wf_nr.time[0], right=wf_hyb.tap_window_start)
        ax3.set_ylim(0, 2)
        ax3.set_xlabel(r't[s]')
        ax3.set_ylabel(r'$Amp ratio$')
        
        # PARAM PLOT
        #ax3 = fig.add_subplot(gs[2, 0]) 
        #ax3.scatter(values_mass1, values_mass2, alpha=0.1, color='black')
        #ax3.plot(value['m1'], value['m2'], 'x', color='red')
        #ax3.set_xlabel(r'$m_1$')
        #ax3.set_ylabel(r'$m_2$')
        #ax4 = fig.add_subplot(gs[2, 1]) 
        #ax4.scatter(values_spin1, values_spin2, alpha=0.1, color='black')
        #ax4.plot(value['s1z'], value['s2z'], 'x', color='red')
        #ax4.set_xlabel(r'$\chi_1$')
        #ax4.set_ylabel(r'$\chi_2$')
        #ax5 = fig.add_subplot(gs[2, 2]) 
        #ax5.scatter(values_lambda1, values_lambda2, alpha=0.1, color='black')
        #ax5.plot(value['lambda1'], value['lambda2'], 'x', color='red')
        #ax5.set_xlabel(r'$\lambda_1$')
        #ax5.set_ylabel(r'$\lambda_2$')
        
        # SAVE PLOT
        if key in yes_dict:
            plt.savefig('../NR_data/td_v4TSurr/yes/'+key[:-3]+".png", dpi=500)
        if key in maybe_dict:
            plt.savefig('../NR_data/td_v4TSurr/maybe/'+key[:-3]+".png", dpi=500)
        if key in short_dict:
            plt.savefig('../NR_data/td_v4TSurr/short/'+key[:-3]+".png", dpi=500)
        if key in lambda5000_dict:
            plt.savefig('../NR_data/td_v4TSurr/lambda5000/'+key[:-3]+".png", dpi=500)
        if key in sacra_dict:
            plt.savefig('../NR_data/td_v4TSurr/sacra/'+key[:-3]+".png", dpi=500)
        if key in doubts_dict:
            plt.savefig('../NR_data/td_v4TSurr/doubts/'+key[:-3]+".png", dpi=500)
        if key in thc_dict:
            plt.savefig('../NR_data/td_v4TSurr/thc/'+key[:-3]+".png", dpi=500)
        #plt.show()
        
        #np.savetxt(basedir_amp+"td_"+str(key[:-3]), np.transpose([wf_hyb.time, delta_amp_td])) 
        #np.savetxt(basedir_phase+"td_"+str(key[:-3]), np.transpose([wf_hyb.time, delta_phase_td])) 
        
        """
        FREQ DOMAIN PLOT
        """
        Mf = np.linspace(0.0004, 0.03, 1000)
        f = Mf/lal.MTSUN_SI/(value['m1']+value['m2'])

        # AMP PLOT
        a1 = compare_wf.interpolate_grid(wf_hyb.freq, wf_hyb.amp_fd, f)
        a2 = compare_wf.interpolate_grid(wf_bns.freq, wf_bns.amp_fd, f)
        delta_amp_fd = a1/a2

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 1)
        #fig.suptitle(key.decode('ascii'), fontsize=16)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.loglog(Mf, a1, label='hyb')
        ax1.loglog(Mf, a2, label='lal')
        ax1.loglog(wf_nr.freq*lal.MTSUN_SI*(value['m1']+value['m2']), wf_nr.amp_fd, label='nr')
        ax1.axvline(wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax1.axvline(1.2*wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax1.axvline(wf_hyb.f_nr_start*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax1.axvline(wf_hyb.f_nr_start_full*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax1.set_xlim(Mf[0], Mf[-1])
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[1, :]) 
        ax2.plot(Mf, delta_amp_fd)
        ax2.axvline(wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax2.axvline(1.2*wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax2.axvline(wf_hyb.f_nr_start*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax2.axvline(wf_hyb.f_nr_start_full*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax2.set_xlim(Mf[0], Mf[-1])

        ax2.set_xscale('log')
        
        # PHASE PLOT 
        p1 = compare_wf.interpolate_grid(wf_hyb.freq, wf_hyb.phase_fd, f)
        p2 = compare_wf.interpolate_grid(wf_bns.freq, wf_bns.phase_fd, f)
        delta_phase_fd = p1-p2

        ax3 = fig.add_subplot(gs[2, :]) 
        ax3.plot(Mf, delta_phase_fd)
        ax3.axvline(wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax3.axvline(1.2*wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax3.axvline(wf_hyb.f_nr_start*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax3.axvline(wf_hyb.f_nr_start_full*lal.MTSUN_SI*(value['m1']+value['m2']))
        ax3.set_xlim(Mf[0], Mf[-1])
        ax3.set_xscale('log')

        if key in yes_dict:
            plt.savefig('../NR_data/fd_v4TSurr/yes/'+key[:-3]+".png", dpi=500)
        if key in maybe_dict:
            plt.savefig('../NR_data/fd_v4TSurr/maybe/'+key[:-3]+".png", dpi=500)
        if key in short_dict:
            plt.savefig('../NR_data/fd_v4TSurr/short/'+key[:-3]+".png", dpi=500)
        if key in lambda5000_dict:
            plt.savefig('../NR_data/fd_v4TSurr/lambda5000/'+key[:-3]+".png", dpi=500)
        if key in sacra_dict:
            plt.savefig('../NR_data/fd_v4TSurr/sacra/'+key[:-3]+".png", dpi=500)
        if key in doubts_dict:
            plt.savefig('../NR_data/fd_v4TSurr/doubts/'+key[:-3]+".png", dpi=500)
        if key in thc_dict:
            plt.savefig('../NR_data/fd_v4TSurr/thc/'+key[:-3]+".png", dpi=500)
        #plt.show()
        
        np.savetxt(basedir_amp+str(key[:-3]), np.transpose([Mf, delta_amp_fd])) 
        np.savetxt(basedir_phase+str(key[:-3]), np.transpose([Mf, delta_phase_fd])) 
